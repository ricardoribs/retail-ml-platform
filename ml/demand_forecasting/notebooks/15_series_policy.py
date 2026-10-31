# Databricks notebook source
# ruff: noqa: E402

# COMMAND ----------

"""Classifies the operational strategy for each product-location series.

This step separates demand profile from numeric forecasting. It uses only history
through the cutoff and persists an auditable contract in
`fct_forecast_series_policy`. The forecast-generation notebook uses this contract
to select a route for each series; model approval still depends on recursive
backtesting."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

if TYPE_CHECKING:
    display: Any
    spark: Any

_REPO_ROOT = next(
    (
        candidate
        for start in (Path.cwd(), *(Path(path) for path in sys.path if path))
        for candidate in (start, *start.parents)
        if (candidate / "pyproject.toml").is_file()
        and (candidate / "ml" / "demand_forecasting").is_dir()
    ),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError("Could not locate the project root in Databricks")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.demand_forecasting.src.config import (
    FORECAST_CONFIG,
    INTERMEDIATE_SCHEMA,
    MARTS_SCHEMA,
    fq_table,
    set_task_value,
)
from ml.demand_forecasting.src.seasonality import (
    MEDIUM_CONFIDENCE_MIN_OBSERVATIONS,
    SEASONALITY_CLASS_LOW,
    SEASONALITY_CLASS_PEAK,
)
from ml.demand_forecasting.src.segmentation import classify_profile

# COMMAND ----------


def show_result(df: DataFrame, rows: int = 100) -> None:
    """Displays a DataFrame in Databricks, with a local fallback."""
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(rows, truncate=False)


FEATURE_MATRIX_TABLE = fq_table(INTERMEDIATE_SCHEMA, "int_forecast_feature_matrix")
SEASONALITY_TABLE = fq_table(MARTS_SCHEMA, "agg_product_seasonality")
POLICY_TABLE = fq_table(MARTS_SCHEMA, "fct_forecast_series_policy")
POLICY_RUN_ID = FORECAST_CONFIG.snapshot_id
POLICY_VERSION = "series_policy_v1"
CUTOFF_MONTH = FORECAST_CONFIG.cutoff_date.replace(day=1)

print("Forecast policy parameters")
print(f"  Matriz de features : {FEATURE_MATRIX_TABLE}")
print(f"  Seasonality        : {SEASONALITY_TABLE}")
print(f"  Output             : {POLICY_TABLE}")
print(f"  Cutoff             : {FORECAST_CONFIG.cutoff_date}")
print(f"  Run ID             : {POLICY_RUN_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Location profile and product seasonality context
# MAGIC
# MAGIC Classification measures recurrence at the product-location grain. The
# MAGIC seasonal signal is aggregated by product, but enables the seasonal route
# MAGIC only when the location has at least three positive months in the rolling
# MAGIC twelve-month window. This prevents aggregate seasonality from converting a
# MAGIC a sparse series into a high-confidence forecast.

# COMMAND ----------

if not spark.catalog.tableExists(SEASONALITY_TABLE):  # noqa: F821
    raise ValueError(
        "Seasonality table not found; run seasonality_ranking "
        "antes de forecast_series_policy."
    )

history_df = (
    spark.table(FEATURE_MATRIX_TABLE)  # noqa: F821
    .filter(F.col("demand_month") <= F.lit(FORECAST_CONFIG.cutoff_date))
    .select("demand_month", "product_variant_id", "location_id", "demand_qty")
)

recency_window = Window.partitionBy("product_variant_id", "location_id").orderBy(
    F.col("demand_month").desc()
)
profile_df = (
    history_df.withColumn("_recency_rank", F.row_number().over(recency_window))
    .groupBy("product_variant_id", "location_id")
    .agg(
        F.countDistinct("demand_month").alias("history_month_count"),
        F.sum(F.when(F.col("demand_qty") > F.lit(0.0), 1).otherwise(0)).alias(
            "positive_demand_month_count"
        ),
        F.sum(
            F.when(
                (F.col("_recency_rank") <= F.lit(12))
                & (F.col("demand_qty") > F.lit(0.0)),
                1,
            ).otherwise(0)
        ).alias("trailing_positive_month_count"),
        F.max(F.when(F.col("demand_qty") > F.lit(0.0), F.col("demand_month"))).alias(
            "last_positive_month"
        ),
    )
    .withColumn(
        "months_since_last_sale",
        F.when(
            F.col("last_positive_month").isNotNull(),
            F.floor(
                F.months_between(
                    F.lit(CUTOFF_MONTH).cast("date"), F.col("last_positive_month")
                )
            ).cast("int"),
        ).otherwise(F.lit(None).cast("int")),
    )
    .drop("last_positive_month")
)

seasonality_context_df = (
    spark.table(SEASONALITY_TABLE)  # noqa: F821
    .groupBy("product_variant_id")
    .agg(
        F.min("observations").cast("int").alias("seasonality_min_observations"),
        F.max(
            F.when(
                F.col("seasonality_class") == F.lit(SEASONALITY_CLASS_PEAK), 1
            ).otherwise(0)
        ).alias("seasonality_has_peak"),
        F.max(
            F.when(
                F.col("seasonality_class") == F.lit(SEASONALITY_CLASS_LOW), 1
            ).otherwise(0)
        ).alias("seasonality_has_low"),
    )
    .withColumn(
        "seasonal_signal_confirmed",
        (
            F.col("seasonality_min_observations")
            >= F.lit(MEDIUM_CONFIDENCE_MIN_OBSERVATIONS)
        )
        & (F.col("seasonality_has_peak") == F.lit(1))
        & (F.col("seasonality_has_low") == F.lit(1)),
    )
    .select(
        "product_variant_id",
        "seasonality_min_observations",
        "seasonal_signal_confirmed",
    )
)

policy_input_rows = (
    profile_df.join(seasonality_context_df, "product_variant_id", "left")
    .fillna(
        {
            "seasonality_min_observations": 0,
            "seasonal_signal_confirmed": False,
        }
    )
    .orderBy("product_variant_id", "location_id")
    .collect()
)
if not policy_input_rows:
    raise ValueError("No demand series are available through the cutoff date")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Per-series policy contract
# MAGIC
# MAGIC `regression_candidate` means only that the series is recurrent; regression
# MAGIC is assigned only if its model family beats the benchmarks in the segment's
# MAGIC recursive backtest. `intermittent_review` does not authorize purchasing.
# MAGIC automatically: its estimate is for planning and requires periodic review.

# COMMAND ----------

created_at = datetime.now(timezone.utc).replace(tzinfo=None)
policy_records = []
for row in policy_input_rows:
    policy = classify_profile(
        history_month_count=int(row["history_month_count"]),
        positive_demand_month_count=int(row["positive_demand_month_count"]),
        trailing_positive_month_count=int(row["trailing_positive_month_count"]),
        months_since_last_sale=(
            int(row["months_since_last_sale"])
            if row["months_since_last_sale"] is not None
            else None
        ),
        seasonal_signal_confirmed=bool(row["seasonal_signal_confirmed"]),
    )
    policy_records.append(
        {
            "run_id": POLICY_RUN_ID,
            "source_cutoff_date": FORECAST_CONFIG.cutoff_date,
            "product_variant_id": row["product_variant_id"],
            "location_id": row["location_id"],
            "demand_segment": policy.demand_segment,
            "forecast_strategy": policy.forecast_strategy,
            "confidence_level": policy.confidence_level,
            "selection_reason": policy.selection_reason,
            "history_month_count": policy.history_month_count,
            "positive_demand_month_count": policy.positive_demand_month_count,
            "trailing_positive_month_count": policy.trailing_positive_month_count,
            "months_since_last_sale": policy.months_since_last_sale,
            "seasonal_signal_confirmed": bool(row["seasonal_signal_confirmed"]),
            "seasonality_min_observations": int(row["seasonality_min_observations"]),
            "policy_version": POLICY_VERSION,
            "created_at": created_at,
        }
    )

policy_df = spark.createDataFrame(policy_records)  # noqa: F821
duplicate_count = (
    policy_df.groupBy("run_id", "product_variant_id", "location_id")
    .count()
    .filter(F.col("count") > F.lit(1))
    .count()
)
if duplicate_count:
    raise AssertionError("The policy contains duplicate product-location keys")
if policy_df.count() != len(policy_input_rows):
    raise AssertionError("The policy does not cover every matrix series")

show_result(
    policy_df.groupBy("demand_segment", "forecast_strategy", "confidence_level")
    .count()
    .orderBy("demand_segment")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Idempotent, traceable persistence
# MAGIC
# MAGIC The key includes the snapshot. Reruns replace only series from the same
# MAGIC run, preserving policies from previous snapshots for auditing.

# COMMAND ----------

if spark.catalog.tableExists(POLICY_TABLE):  # noqa: F821
    from delta.tables import DeltaTable

    target = DeltaTable.forName(spark, POLICY_TABLE)  # noqa: F821
    merge_condition = (
        "target.run_id = source.run_id "
        "AND target.product_variant_id = source.product_variant_id "
        "AND target.location_id = source.location_id"
    )
    (
        target.alias("target")
        .merge(policy_df.alias("source"), merge_condition)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .whenNotMatchedBySourceDelete(condition=f"target.run_id = '{POLICY_RUN_ID}'")
        .execute()
    )
else:
    policy_df.write.format("delta").mode("append").saveAsTable(POLICY_TABLE)

persisted_policy_df = spark.table(POLICY_TABLE).filter(  # noqa: F821
    F.col("run_id") == F.lit(POLICY_RUN_ID)
)
persisted_count = persisted_policy_df.count()
if persisted_count != len(policy_records):
    raise AssertionError(
        "Policy persistence is incomplete: "
        f"expected={len(policy_records)}, found={persisted_count}"
    )

print(f"Policy persisted to: {POLICY_TABLE}")
print(f"Series classified: {persisted_count}")
set_task_value("policy_run_id", POLICY_RUN_ID)
set_task_value("policy_refreshed", True)
print("SERIES POLICY COMPLETE")
