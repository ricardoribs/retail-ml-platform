# Databricks notebook source
# ruff: noqa: E402

# COMMAND ----------

"""Measures recursive baselines for each demand profile.

The profile is recalculated at the end of training without consulting the evaluated
horizon. Persisted metrics are the evidence used to approve or reject the seasonal
route by product-location. This step does not publish forecasts or change demand data."""

from __future__ import annotations

import math
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

from ml.demand_forecasting.src.baseline import evaluate_recursive_baselines
from ml.demand_forecasting.src.config import (
    FORECAST_CONFIG,
    INTERMEDIATE_SCHEMA,
    MARTS_SCHEMA,
    fq_table,
    set_task_value,
)
from ml.demand_forecasting.src.modeling import (
    LABEL_COLUMN,
    SERIES_KEY_COLUMNS,
    build_temporal_split,
    collect_backtest_history,
    shift_months,
)
from ml.demand_forecasting.src.seasonality import (
    MEDIUM_CONFIDENCE_MIN_OBSERVATIONS,
    SEASONAL_LOW_INDEX_THRESHOLD,
    SEASONAL_PEAK_INDEX_THRESHOLD,
)
from ml.demand_forecasting.src.segmentation import classify_profile

# COMMAND ----------


def show_result(df: DataFrame, rows: int = 100) -> None:
    """Displays a DataFrame in Databricks, with a local fallback."""
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(rows, truncate=False)


def _nullable_metric(value: float | int | str | None) -> float | None:
    """Convert undefined metrics to NULL for Delta persistence."""
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


FEATURE_MATRIX_TABLE = fq_table(INTERMEDIATE_SCHEMA, "int_forecast_feature_matrix")
SEGMENT_METRICS_TABLE = fq_table(MARTS_SCHEMA, "fct_forecast_segment_metrics")
TEST_WINDOW_MONTHS = len(FORECAST_CONFIG.horizon_months)
TEMPORAL_SPLIT = build_temporal_split(
    FORECAST_CONFIG.cutoff_date,
    TEST_WINDOW_MONTHS,
)
TARGET_MONTHS = tuple(
    shift_months(TEMPORAL_SPLIT.test_start, offset)
    for offset in range(TEST_WINDOW_MONTHS)
)
MODEL_RUN_ID = f"model_{FORECAST_CONFIG.snapshot_id}"
METRICS_VERSION = "segment_baseline_v1"

print("Segmented backtest parameters")
print(f"  Matriz de features : {FEATURE_MATRIX_TABLE}")
print(f"  Output             : {SEGMENT_METRICS_TABLE}")
print(f"  Training end       : {TEMPORAL_SPLIT.train_end}")
print(f"  Teste              : {TEMPORAL_SPLIT.test_start} a {TEMPORAL_SPLIT.test_end}")
print(f"  Run ID             : {MODEL_RUN_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. As-of profile and leakage-free seasonality
# MAGIC
# MAGIC Both local recurrence and the product seasonality signal are recalculated
# MAGIC using data only through `train_end`. No sales from the test window are used.
# MAGIC determines the segment or strategy being evaluated.

# COMMAND ----------

feature_matrix_df = spark.table(FEATURE_MATRIX_TABLE)  # noqa: F821
history_df = feature_matrix_df.filter(
    F.col("demand_month") <= F.lit(TEMPORAL_SPLIT.train_end)
)
test_df = feature_matrix_df.filter(
    (F.col("demand_month") >= F.lit(TEMPORAL_SPLIT.test_start))
    & (F.col("demand_month") <= F.lit(TEMPORAL_SPLIT.test_end))
)

recency_window = Window.partitionBy(*SERIES_KEY_COLUMNS).orderBy(
    F.col("demand_month").desc()
)
profile_df = (
    history_df.select(
        "demand_month", *SERIES_KEY_COLUMNS, F.col(LABEL_COLUMN).alias("demand_qty")
    )
    .withColumn("_recency_rank", F.row_number().over(recency_window))
    .groupBy(*SERIES_KEY_COLUMNS)
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
                    F.lit(TEMPORAL_SPLIT.train_end.replace(day=1)).cast("date"),
                    F.col("last_positive_month"),
                )
            ).cast("int"),
        ).otherwise(F.lit(None).cast("int")),
    )
    .drop("last_positive_month")
)

product_month_df = (
    history_df.groupBy("product_variant_id", "demand_month")
    .agg(F.sum(F.col(LABEL_COLUMN).cast("double")).alias("demand_qty"))
    .withColumn("month_number", F.month("demand_month"))
    .withColumn("year_number", F.year("demand_month"))
)
monthly_average_df = product_month_df.groupBy("product_variant_id", "month_number").agg(
    F.avg("demand_qty").alias("avg_monthly_demand_qty"),
    F.countDistinct("year_number").alias("observations"),
)
product_seasonality_df = (
    monthly_average_df.groupBy("product_variant_id")
    .agg(
        F.avg("avg_monthly_demand_qty").alias("overall_avg_monthly_demand_qty"),
        F.min("observations").cast("int").alias("seasonality_min_observations"),
    )
    .join(monthly_average_df, "product_variant_id", "inner")
    .withColumn(
        "seasonality_index",
        F.when(
            F.col("overall_avg_monthly_demand_qty") > F.lit(0.0),
            F.col("avg_monthly_demand_qty") / F.col("overall_avg_monthly_demand_qty"),
        ),
    )
    .groupBy("product_variant_id", "seasonality_min_observations")
    .agg(
        F.max(
            F.when(
                F.col("seasonality_index") >= F.lit(SEASONAL_PEAK_INDEX_THRESHOLD),
                1,
            ).otherwise(0)
        ).alias("seasonality_has_peak"),
        F.max(
            F.when(
                F.col("seasonality_index") <= F.lit(SEASONAL_LOW_INDEX_THRESHOLD),
                1,
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
    .select("product_variant_id", "seasonal_signal_confirmed")
)

profile_rows = (
    profile_df.join(product_seasonality_df, "product_variant_id", "left")
    .fillna({"seasonal_signal_confirmed": False})
    .collect()
)
segment_by_key = {}
for row in profile_rows:
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
    segment_by_key[(row["product_variant_id"], row["location_id"])] = (
        policy.demand_segment
    )

if not segment_by_key:
    raise ValueError("No series are available for backtesting")

segment_count_records = [
    {
        "demand_segment": segment,
        "series_count": list(segment_by_key.values()).count(segment),
    }
    for segment in sorted(set(segment_by_key.values()))
]
show_result(spark.createDataFrame(segment_count_records).orderBy("demand_segment"))  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Baselines over the recursive horizon by segment
# MAGIC
# MAGIC Forecasts for months 2 and 3 use the previous month's output in their own
# MAGIC series. The seasonal path will be approved later only if it outperforms the
# MAGIC zero-reference baseline on this same slice.

# COMMAND ----------

test_keys = {
    (row["product_variant_id"], row["location_id"])
    for row in test_df.select(*SERIES_KEY_COLUMNS).distinct().collect()
}
missing_segments = test_keys.difference(segment_by_key)
if missing_segments:
    raise AssertionError("Some test series have no as-of profile")

history_states = collect_backtest_history(
    feature_matrix_df,
    test_df,
    TEMPORAL_SPLIT.train_end,
)
actual_rows = test_df.select(
    "demand_month", *SERIES_KEY_COLUMNS, LABEL_COLUMN
).collect()

created_at = datetime.now(timezone.utc).replace(tzinfo=None)
metric_records = []
for demand_segment in sorted(set(segment_by_key.values())):
    segment_keys = {
        key for key, segment in segment_by_key.items() if segment == demand_segment
    }
    segment_states = {
        key: state for key, state in history_states.items() if key in segment_keys
    }
    segment_actual_rows = [
        row
        for row in actual_rows
        if (row["product_variant_id"], row["location_id"]) in segment_keys
    ]
    if not segment_actual_rows:
        continue

    baseline_metrics = evaluate_recursive_baselines(
        segment_states,
        TARGET_MONTHS,
        segment_actual_rows,
    )
    finite_wape_rows = [
        row
        for row in baseline_metrics.values()
        if _nullable_metric(row["wape"]) is not None
    ]
    best_model_name = (
        min(
            finite_wape_rows,
            key=lambda row: (float(row["wape"]), float(row["mae"]), row["model_name"]),
        )["model_name"]
        if finite_wape_rows
        else None
    )
    for model_name, metrics in baseline_metrics.items():
        metric_records.append(
            {
                "run_id": MODEL_RUN_ID,
                "demand_segment": demand_segment,
                "model_name": model_name,
                "model_version": METRICS_VERSION,
                "train_end_month": TEMPORAL_SPLIT.train_end,
                "test_start_month": TEMPORAL_SPLIT.test_start,
                "test_end_month": TEMPORAL_SPLIT.test_end,
                "mae": _nullable_metric(metrics["mae"]),
                "rmse": _nullable_metric(metrics["rmse"]),
                "mape": _nullable_metric(metrics["mape"]),
                "wape": _nullable_metric(metrics["wape"]),
                "evaluated_rows": int(metrics["evaluated_rows"]),
                "fallback_rows": int(metrics["fallback_rows"]),
                "is_best_model": model_name == best_model_name,
                "created_at": created_at,
            }
        )

if not metric_records:
    raise ValueError("The segmented backtest produced no metrics")

metrics_df = spark.createDataFrame(metric_records)  # noqa: F821
show_result(metrics_df.orderBy("demand_segment", "wape", "model_name"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Idempotent persistence
# MAGIC
# MAGIC The key includes the run, segment, and model. A run's metric set
# MAGIC is fully replaced on rerun, preserving previous runs.

# COMMAND ----------

if spark.catalog.tableExists(SEGMENT_METRICS_TABLE):  # noqa: F821
    from delta.tables import DeltaTable

    target = DeltaTable.forName(spark, SEGMENT_METRICS_TABLE)  # noqa: F821
    merge_condition = (
        "target.run_id = source.run_id "
        "AND target.demand_segment = source.demand_segment "
        "AND target.model_name = source.model_name"
    )
    (
        target.alias("target")
        .merge(metrics_df.alias("source"), merge_condition)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .whenNotMatchedBySourceDelete(condition=f"target.run_id = '{MODEL_RUN_ID}'")
        .execute()
    )
else:
    metrics_df.write.format("delta").mode("append").saveAsTable(SEGMENT_METRICS_TABLE)

persisted_metrics_df = spark.table(SEGMENT_METRICS_TABLE).filter(  # noqa: F821
    F.col("run_id") == F.lit(MODEL_RUN_ID)
)
if persisted_metrics_df.count() != len(metric_records):
    raise AssertionError("Segmented metric persistence is incomplete")

print(f"Segmented metrics persisted to: {SEGMENT_METRICS_TABLE}")
set_task_value("segment_metrics_run_id", MODEL_RUN_ID)
set_task_value("segment_metrics_refreshed", True)
print("SEGMENTED BACKTEST COMPLETED")
