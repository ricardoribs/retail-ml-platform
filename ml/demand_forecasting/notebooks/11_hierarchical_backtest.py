# Databricks notebook source
# ruff: noqa: E402

"""Exploratory hierarchical backtest; read-only and outside the daily job.

Compares forecasts by category/location, allocated to product-location series, in
the two previously audited holdouts. History and weights end before the first
forecast month; actuals are used only for metrics."""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

if TYPE_CHECKING:
    dbutils: Any
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
    fq_table,
)
from ml.demand_forecasting.src.hierarchical import (
    build_hierarchical_forecasts,
    evaluate_forecasts,
)
from ml.demand_forecasting.src.modeling import (
    LABEL_COLUMN,
    SERIES_KEY_COLUMNS,
    build_temporal_split,
    prepare_temporal_frames,
    shift_months,
)


def show_result(df: DataFrame) -> None:
    """Display only the small result tables."""
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(100, truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Hierarchical backtest: aggregate grain versus product/location
# MAGIC
# MAGIC Each cutoff uses the same eligible cohort as operational training and
# MAGIC forecasts the next three months without reading sales from those months
# MAGIC when building the forecast. Candidates are fixed rules, not selected using
# MAGIC the holdout.
# MAGIC This notebook does not write tables, update the published forecast, or
# MAGIC affects the operational gate.

# COMMAND ----------

dbutils.widgets.text("cutoff_dates", "2025-11-30,2026-08-31")  # noqa: F821
cutoff_dates = tuple(
    date.fromisoformat(value.strip())
    for value in dbutils.widgets.get("cutoff_dates").split(",")  # noqa: F821
    if value.strip()
)
if not cutoff_dates or len(cutoff_dates) != len(set(cutoff_dates)):
    raise ValueError("Provide distinct cutoff dates in YYYY-MM-DD format")
for cutoff in cutoff_dates:
    if cutoff != shift_months(cutoff, 1) - timedelta(days=1):
        raise ValueError(f"The cutoff must be the last day of the month: {cutoff}")
    if cutoff > FORECAST_CONFIG.cutoff_date:
        raise ValueError(f"Cutoff is after the last closed month: {cutoff}")

feature_table = fq_table(INTERMEDIATE_SCHEMA, "int_forecast_feature_matrix")
feature_matrix_df = spark.table(feature_table).filter(  # noqa: F821
    F.col("demand_month") <= F.lit(max(cutoff_dates))
)
duplicate_keys = (
    feature_matrix_df.groupBy("demand_month", *SERIES_KEY_COLUMNS)
    .count()
    .filter(F.col("count") != 1)
    .count()
)
invalid_demands = feature_matrix_df.filter(
    F.col(LABEL_COLUMN).isNull() | (F.col(LABEL_COLUMN) < 0)
).count()
if duplicate_keys or invalid_demands:
    raise ValueError(
        f"Invalid source: {duplicate_keys} duplicate keys and "
        f"{invalid_demands} demandas nulas/negativas"
    )
summary_rows: list[dict[str, float | int | str]] = []
month_rows: list[dict[str, float | int | str]] = []
location_type_rows: list[dict[str, float | int | str]] = []

for cutoff in cutoff_dates:
    split = build_temporal_split(cutoff, test_months=3)
    target_months = tuple(shift_months(split.test_start, i) for i in range(3))
    eligible_df, _, test_df = prepare_temporal_frames(
        feature_matrix_df=feature_matrix_df,
        split=split,
        min_history_months=FORECAST_CONFIG.min_history_months,
        min_positive_demand_months=FORECAST_CONFIG.min_positive_demand_months,
    )
    history_df = feature_matrix_df.filter(
        F.col("demand_month") <= F.lit(split.train_end)
    ).join(
        eligible_df.select(*SERIES_KEY_COLUMNS),
        list(SERIES_KEY_COLUMNS),
        "inner",
    )

    # The category mapping comes from the last record BEFORE the test.
    latest = Window.partitionBy(*SERIES_KEY_COLUMNS).orderBy(
        F.col("demand_month").desc()
    )
    series_group_df = (
        history_df.withColumn("_rank", F.row_number().over(latest))
        .filter(F.col("_rank") == 1)
        .select(
            *SERIES_KEY_COLUMNS,
            F.coalesce(F.col("category_name"), F.lit("__missing__")).alias("_category"),
            F.coalesce(F.col("location_type"), F.lit("__missing__")).alias(
                "_location_type"
            ),
        )
    )
    series_metadata_rows = series_group_df.collect()
    series_groups = {
        (row["product_variant_id"], row["location_id"]): (
            row["location_id"],
            row["_category"],
        )
        for row in series_metadata_rows
    }
    series_types = {
        (row["product_variant_id"], row["location_id"]): row["_location_type"]
        for row in series_metadata_rows
    }
    if (
        len(series_metadata_rows) != len(series_groups)
        or len(series_groups) != eligible_df.count()
    ):
        raise ValueError("The category mapping does not cover the eligible cohort")

    # Reassign each historical month to the category known at the cutoff to avoid
    # group changes within a series; do not read the category from the holdout.
    group_month_df = (
        history_df.drop("category_name")
        .join(series_group_df, list(SERIES_KEY_COLUMNS), "inner")
        .groupBy("demand_month", "location_id", "_category")
        .agg(F.sum(LABEL_COLUMN).alias("quantity"))
    )
    group_history = defaultdict(dict)
    for row in group_month_df.collect():
        group_history[(row["location_id"], row["_category"])][row["demand_month"]] = (
            float(row["quantity"])
        )

    last12_start = shift_months(split.test_start, -12)
    recent_df = history_df.filter(F.col("demand_month") >= F.lit(last12_start))
    local_last12 = {
        (row["product_variant_id"], row["location_id"]): float(row["quantity"])
        for row in recent_df.groupBy(*SERIES_KEY_COLUMNS)
        .agg(F.sum(LABEL_COLUMN).alias("quantity"))
        .collect()
    }
    pooled_last12 = {
        row["product_variant_id"]: float(row["quantity"])
        for row in recent_df.groupBy("product_variant_id")
        .agg(F.sum(LABEL_COLUMN).alias("quantity"))
        .collect()
    }

    actual_rows = test_df.select(
        "demand_month", *SERIES_KEY_COLUMNS, LABEL_COLUMN
    ).collect()
    actual = {
        (row["demand_month"], row["product_variant_id"], row["location_id"]): float(
            row[LABEL_COLUMN]
        )
        for row in actual_rows
    }
    expected_keys = {
        (month, product_id, location_id)
        for month in target_months
        for product_id, location_id in series_groups
    }
    if len(actual_rows) != len(actual) or set(actual) != expected_keys:
        raise ValueError(
            f"Holdout {cutoff}: series/months are missing or keys are duplicated"
        )

    forecasts = build_hierarchical_forecasts(
        series_groups=series_groups,
        group_history=group_history,
        local_last12=local_last12,
        pooled_last12=pooled_last12,
        target_months=target_months,
    )
    forecasts["zero"] = dict.fromkeys(actual, 0.0)
    for method, predicted in forecasts.items():
        metrics = evaluate_forecasts(actual, predicted, series_groups)
        summary_rows.append(
            {
                "cutoff": cutoff.isoformat(),
                "method": method,
                "series": len(series_groups),
                "rows": len(actual),
                "actual_total": round(sum(actual.values()), 2),
                "predicted_total": round(sum(predicted.values()), 2),
                "sku_location_wape": round(metrics["sku_location"]["wape"], 2),
                "sku_location_mae": round(metrics["sku_location"]["mae"], 3),
                "category_location_wape": round(
                    metrics["category_location"]["wape"], 2
                ),
                "location_wape": round(metrics["location"]["wape"], 2),
                "total_month_wape": round(metrics["total_month"]["wape"], 2),
            }
        )
        for location_type in sorted(set(series_types.values())):
            segment_actual = {
                key: quantity
                for key, quantity in actual.items()
                if series_types[key[1:]] == location_type
            }
            segment_predicted = {key: predicted[key] for key in segment_actual}
            segment_groups = {
                series: group
                for series, group in series_groups.items()
                if series_types[series] == location_type
            }
            segment_metrics = evaluate_forecasts(
                segment_actual, segment_predicted, segment_groups
            )["sku_location"]
            location_type_rows.append(
                {
                    "cutoff": cutoff.isoformat(),
                    "method": method,
                    "location_type": location_type,
                    "series": len(segment_groups),
                    "actual_total": round(sum(segment_actual.values()), 2),
                    "predicted_total": round(sum(segment_predicted.values()), 2),
                    "sku_location_wape": round(segment_metrics["wape"], 2),
                    "sku_location_mae": round(segment_metrics["mae"], 3),
                }
            )
        for month in target_months:
            month_keys = [key for key in actual if key[0] == month]
            month_rows.append(
                {
                    "cutoff": cutoff.isoformat(),
                    "method": method,
                    "month": month.isoformat(),
                    "actual_total": round(sum(actual[key] for key in month_keys), 2),
                    "predicted_total": round(
                        sum(predicted[key] for key in month_keys), 2
                    ),
                }
            )

    print(
        f"Cutoff {cutoff}: {len(series_groups)} series, {len(actual)} rows; "
        "forecast uses only history from before the holdout"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results
# MAGIC
# MAGIC The primary criterion is **sku_location_wape**: improving only the
# MAGIC aggregate does not solve the product-by-store forecast requirement.
# MAGIC Compare with `zero` (WAPE is 100% when there is demand). The 2025 window
# MAGIC must also pass; do not choose a method based only on 2026. Also compare
# MAGIC against the per-series benchmarks from notebook 09 at the same cutoff.

# COMMAND ----------

show_result(spark.createDataFrame(summary_rows).orderBy("cutoff", "method"))  # noqa: F821
show_result(
    spark.createDataFrame(location_type_rows).orderBy(  # noqa: F821
        "cutoff", "location_type", "method"
    )
)
show_result(spark.createDataFrame(month_rows).orderBy("cutoff", "method", "month"))  # noqa: F821
