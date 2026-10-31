# Databricks notebook source
# ruff: noqa: E402

# COMMAND ----------

"""Calculates and persists forecast growth by store/POS.

Compares realized demand in the three closed months through the cutoff with the
three forecast horizons from the latest complete run. Growth percentage is null
when the realized baseline is zero."""

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
    upstream_task_value,
)
from ml.demand_forecasting.src.modeling import shift_months

# COMMAND ----------


def show_result(df: DataFrame, rows: int = 100) -> None:
    """Displays a DataFrame in Databricks, with a local fallback."""
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(rows, truncate=False)


MONTHLY_DEMAND_TABLE = fq_table(
    INTERMEDIATE_SCHEMA,
    "int_monthly_product_location_demand",
)
FORECAST_TABLE = fq_table(MARTS_SCHEMA, "agg_location_demand_forecast")
LOCATION_GROWTH_TABLE = fq_table(MARTS_SCHEMA, "agg_location_forecast_growth")
FORECAST_RUN_ID = upstream_task_value(
    "location_aggregate_forecast", "snapshot_id", FORECAST_CONFIG.snapshot_id
)
if FORECAST_RUN_ID != FORECAST_CONFIG.snapshot_id:
    raise ValueError("The received snapshot does not match the execution cutoff")

COMPARISON_TYPE = "store"
COMPARISON_MONTHS = len(FORECAST_CONFIG.horizon_months)
ACTUAL_END_DATE = FORECAST_CONFIG.cutoff_date
ACTUAL_START_MONTH = shift_months(
    ACTUAL_END_DATE.replace(day=1),
    -(COMPARISON_MONTHS - 1),
)

print("Store/POS growth parameters")
print(f"  Monthly demand: {MONTHLY_DEMAND_TABLE}")
print(f"  Forecast: {FORECAST_TABLE}")
print(f"  Output: {LOCATION_GROWTH_TABLE}")
print(f"  Realizado: {ACTUAL_START_MONTH} a {ACTUAL_END_DATE}")
print(f"  Forecast horizons: 1 to {COMPARISON_MONTHS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Select the snapshot produced by this run
# MAGIC
# MAGIC The analysis uses the `snapshot_id` passed from the forecast task and
# MAGIC requires exactly horizons 1, 2, and 3.

# COMMAND ----------

forecast_history_df = spark.table(FORECAST_TABLE)  # noqa: F821

selected_forecast_df = forecast_history_df.filter(
    F.col("run_id") == F.lit(FORECAST_RUN_ID)
)
selected_run_summary = selected_forecast_df.agg(
    F.countDistinct("horizon_month").alias("horizon_count"),
    F.min("horizon_month").alias("min_horizon"),
    F.max("horizon_month").alias("max_horizon"),
).first()

if (
    selected_run_summary is None
    or selected_run_summary["horizon_count"] != COMPARISON_MONTHS
    or selected_run_summary["min_horizon"] != 1
    or selected_run_summary["max_horizon"] != COMPARISON_MONTHS
):
    raise ValueError(f"Snapshot {FORECAST_RUN_ID} does not contain all three horizons")

incomplete_location_count = (
    selected_forecast_df.groupBy("location_id")
    .agg(F.countDistinct("horizon_month").alias("horizon_count"))
    .filter(F.col("horizon_count") != F.lit(COMPARISON_MONTHS))
    .count()
)
if incomplete_location_count:
    raise ValueError(
        f"Forecast run {FORECAST_RUN_ID} contains "
        f"{incomplete_location_count} locations have an incomplete horizon"
    )

print(f"Selected forecast run: {FORECAST_RUN_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Actual and forecast demand by store

# COMMAND ----------

actual_by_location_df = (
    spark.table(MONTHLY_DEMAND_TABLE)  # noqa: F821
    .filter(
        (F.col("location_type") == F.lit(COMPARISON_TYPE))
        & (F.col("demand_month") >= F.lit(ACTUAL_START_MONTH))
        & (F.col("demand_month") <= F.lit(ACTUAL_END_DATE))
    )
    .groupBy("location_id")
    .agg(
        F.sum(F.col("demand_qty").cast("double")).alias("actual_last_3m_demand_qty"),
        F.countDistinct("demand_month").alias("actual_month_count"),
    )
)

forecast_by_location_df = (
    selected_forecast_df.filter(
        (F.col("location_type") == F.lit(COMPARISON_TYPE))
        & F.col("horizon_month").between(1, COMPARISON_MONTHS)
    )
    .groupBy("location_id")
    .agg(
        F.max("location_key").alias("location_key"),
        F.max("location_name").alias("location_name"),
        F.max("location_type").alias("location_type"),
        F.max("city").alias("city"),
        F.max("state").alias("state"),
        F.min("forecast_month").alias("forecast_start_month"),
        F.max("forecast_month").alias("forecast_end_month"),
        F.sum(F.col("forecast_demand_qty").cast("double")).alias(
            "forecast_next_3m_demand_qty"
        ),
        F.countDistinct("horizon_month").alias("forecast_horizon_count"),
    )
)

invalid_location_count = forecast_by_location_df.filter(
    F.col("forecast_horizon_count") != F.lit(COMPARISON_MONTHS)
).count()
if invalid_location_count:
    raise ValueError(
        f"{invalid_location_count} stores do not have all three forecast horizons"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Crescimento e rankings
# MAGIC
# MAGIC The percentage and its ranking require actuals for all three months. When
# MAGIC less history is available, the row receives `insufficient_actual_history`.
# MAGIC When
# MAGIC all three months exist but their sum is zero, it receives `new_or_inactive_base`.

# COMMAND ----------

growth_base_df = (
    forecast_by_location_df.join(actual_by_location_df, "location_id", "left")
    .fillna(
        {
            "actual_last_3m_demand_qty": 0.0,
            "actual_month_count": 0,
        }
    )
    .withColumn(
        "growth_abs_qty",
        F.col("forecast_next_3m_demand_qty") - F.col("actual_last_3m_demand_qty"),
    )
    .withColumn(
        "growth_pct",
        F.when(
            (F.col("actual_month_count") == F.lit(COMPARISON_MONTHS))
            & (F.col("actual_last_3m_demand_qty") > F.lit(0.0)),
            F.col("growth_abs_qty") / F.col("actual_last_3m_demand_qty") * F.lit(100.0),
        ).otherwise(F.lit(None).cast("double")),
    )
    .withColumn(
        "growth_base_status",
        F.when(
            F.col("actual_month_count") < F.lit(COMPARISON_MONTHS),
            F.lit("insufficient_actual_history"),
        )
        .when(
            F.col("actual_last_3m_demand_qty") == F.lit(0.0),
            F.lit("new_or_inactive_base"),
        )
        .otherwise(F.lit("comparable")),
    )
)

growth_rank_abs_window = Window.orderBy(F.col("growth_abs_qty").desc())
growth_rank_pct_window = Window.orderBy(F.col("growth_pct").desc_nulls_last())
created_at = datetime.now(timezone.utc).replace(tzinfo=None)

location_growth_df = (
    growth_base_df.withColumn(
        "growth_rank_abs",
        F.rank().over(growth_rank_abs_window),
    )
    .withColumn(
        "growth_rank_pct",
        F.when(
            F.col("growth_pct").isNotNull(),
            F.rank().over(growth_rank_pct_window),
        ).otherwise(F.lit(None).cast("integer")),
    )
    .select(
        "location_id",
        "location_key",
        "location_name",
        "location_type",
        "city",
        "state",
        F.lit(COMPARISON_TYPE).alias("comparison_type"),
        F.lit(FORECAST_RUN_ID).alias("source_forecast_run_id"),
        F.lit(ACTUAL_START_MONTH).cast("date").alias("actual_start_month"),
        F.lit(ACTUAL_END_DATE).cast("date").alias("actual_end_date"),
        "forecast_start_month",
        "forecast_end_month",
        "actual_last_3m_demand_qty",
        "forecast_next_3m_demand_qty",
        "growth_abs_qty",
        "growth_pct",
        "growth_rank_abs",
        "growth_rank_pct",
        "growth_base_status",
        "actual_month_count",
        "forecast_horizon_count",
        F.lit(created_at).cast("timestamp").alias("created_at"),
    )
)

location_count = location_growth_df.count()
if location_count == 0:
    raise ValueError("No stores were found for the growth calculation")

show_result(location_growth_df.orderBy("growth_rank_abs", "location_id"))
print(f"Ranked stores/POS: {location_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Persist the current view
# MAGIC
# MAGIC The `MERGE` updates the store snapshot using the key
# MAGIC `location_id + comparison_type`. Old `store` rows are removed,
# MAGIC without deleting records for other types, such as `warehouse`.

# COMMAND ----------

if spark.catalog.tableExists(LOCATION_GROWTH_TABLE):  # noqa: F821
    from delta.tables import DeltaTable

    target = DeltaTable.forName(spark, LOCATION_GROWTH_TABLE)  # noqa: F821
    merge_condition = (
        "target.location_id = source.location_id "
        "AND target.comparison_type = source.comparison_type"
    )
    (
        target.alias("target")
        .merge(location_growth_df.alias("source"), merge_condition)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .whenNotMatchedBySourceDelete(
            condition=f"target.comparison_type = '{COMPARISON_TYPE}'"
        )
        .execute()
    )
else:
    # Append also creates the table on the first run and works in Serverless.
    location_growth_df.write.format("delta").mode("append").saveAsTable(
        LOCATION_GROWTH_TABLE
    )

persisted_growth_df = spark.table(LOCATION_GROWTH_TABLE)  # noqa: F821
show_result(
    persisted_growth_df.select(
        "location_name",
        "location_type",
        "actual_last_3m_demand_qty",
        "forecast_next_3m_demand_qty",
        "growth_abs_qty",
        "growth_pct",
        "growth_rank_abs",
    ).orderBy("growth_rank_abs", "location_id")
)

print(f"Ranking persisted to: {LOCATION_GROWTH_TABLE}")
print(f"Source forecast run: {FORECAST_RUN_ID}")
print("COMPLETED")
