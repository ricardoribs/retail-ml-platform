# Databricks notebook source
# ruff: noqa: E402

"""Backtests a two-stage forecast: location to product-location.

This experiment addresses a portfolio warning signal: the current granular model
may correctly identify an individual series as intermittent but zero too much volume
when results are aggregated. The first stage forecasts each store/DC with seasonality;
the second allocates the total to products using as-of shares.

This notebook is read-only. It does not write tables, change the published forecast,
or approve rankings. It evaluates three temporal folds before any promotion decision."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyspark.sql import DataFrame
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
from ml.demand_forecasting.src.hierarchical import evaluate_forecasts
from ml.demand_forecasting.src.location_allocation import (
    LOCATION_ALLOCATION_METHODS,
    build_location_allocation_forecasts,
)
from ml.demand_forecasting.src.modeling import (
    LABEL_COLUMN,
    SERIES_KEY_COLUMNS,
    build_temporal_split,
    prepare_temporal_frames,
    shift_months,
)


def show_result(df: DataFrame) -> None:
    """Display a table in either a notebook or a job."""
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(200, truncate=False)


def month_end(month_start):
    return shift_months(month_start, 1) - timedelta(days=1)


# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation protocol
# MAGIC
# MAGIC The three cutoffs match operational selection. In each fold, history ends
# MAGIC before the first month being evaluated. Actuals are used only to measure
# MAGIC WAPE and bias; no parameters are selected based on holdout results.

# COMMAND ----------

TEST_WINDOW_MONTHS = len(FORECAST_CONFIG.horizon_months)
FOLD_OFFSETS = (0, TEST_WINDOW_MONTHS, TEST_WINDOW_MONTHS * 2)
FOLD_CUTOFFS = tuple(
    month_end(shift_months(FORECAST_CONFIG.cutoff_date, -offset))
    for offset in FOLD_OFFSETS
)
FEATURE_MATRIX_TABLE = fq_table(INTERMEDIATE_SCHEMA, "int_forecast_feature_matrix")

print("Location-to-product-location backtest")
print(f"  Source: {FEATURE_MATRIX_TABLE}")
print(f"  Cortes: {', '.join(str(cutoff) for cutoff in FOLD_CUTOFFS)}")
print("  Persistence: none (read-only experiment)")

feature_matrix_df = spark.table(FEATURE_MATRIX_TABLE)  # noqa: F821
duplicate_keys = (
    feature_matrix_df.groupBy("demand_month", *SERIES_KEY_COLUMNS)
    .count()
    .filter(F.col("count") != F.lit(1))
    .count()
)
invalid_demand = feature_matrix_df.filter(
    F.col(LABEL_COLUMN).isNull() | (F.col(LABEL_COLUMN) < F.lit(0))
).count()
if duplicate_keys or invalid_demand:
    raise ValueError(
        f"Invalid matrix: {duplicate_keys} duplicate keys and "
        f"{invalid_demand} demandas nulas/negativas"
    )

metric_rows: list[dict[str, object]] = []

for fold_index, cutoff in enumerate(FOLD_CUTOFFS, start=1):
    split = build_temporal_split(cutoff, TEST_WINDOW_MONTHS)
    target_months = tuple(
        shift_months(split.test_start, offset) for offset in range(TEST_WINDOW_MONTHS)
    )
    eligible_df, _, test_df = prepare_temporal_frames(
        feature_matrix_df=feature_matrix_df,
        split=split,
        min_history_months=FORECAST_CONFIG.min_history_months,
        min_positive_demand_months=FORECAST_CONFIG.min_positive_demand_months,
    )
    history_df = feature_matrix_df.filter(
        F.col("demand_month") <= F.lit(split.train_end)
    ).join(eligible_df.select(*SERIES_KEY_COLUMNS), list(SERIES_KEY_COLUMNS), "inner")

    series_history: dict[tuple[int, int], dict] = defaultdict(dict)
    for row in history_df.select(
        "demand_month", *SERIES_KEY_COLUMNS, LABEL_COLUMN
    ).collect():
        series_history[(row["product_variant_id"], row["location_id"])][
            row["demand_month"]
        ] = float(row[LABEL_COLUMN])

    series_groups = {
        (product_id, location_id): (location_id, "__location_total__")
        for product_id, location_id in series_history
    }
    expected_series_count = eligible_df.count()
    if len(series_history) != expected_series_count:
        raise ValueError("History does not exactly cover the eligible cohort")

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
        for product_id, location_id in series_history
    }
    if len(actual) != len(actual_rows) or set(actual) != expected_keys:
        raise ValueError(f"Fold {fold_index}: actuals do not cover the cohort")

    predictions = build_location_allocation_forecasts(
        series_history=series_history,
        target_months=target_months,
    )
    predictions["zero"] = dict.fromkeys(actual, 0.0)
    for method in ("zero", *LOCATION_ALLOCATION_METHODS):
        predicted = predictions[method]
        metrics = evaluate_forecasts(actual, predicted, series_groups)
        actual_total = sum(actual.values())
        predicted_total = sum(predicted.values())
        metric_rows.append(
            {
                "fold_index": fold_index,
                "cutoff": cutoff.isoformat(),
                "method": method,
                "series": len(series_history),
                "rows": len(actual),
                "actual_total": round(actual_total, 2),
                "predicted_total": round(predicted_total, 2),
                "volume_bias_pct": round(
                    (predicted_total - actual_total) / actual_total * 100.0, 2
                )
                if actual_total > 0
                else None,
                "sku_location_wape": round(metrics["sku_location"]["wape"], 2),
                "location_wape": round(metrics["location"]["wape"], 2),
                "total_month_wape": round(metrics["total_month"]["wape"], 2),
            }
        )
    print(
        f"Fold {fold_index}: training through {split.train_end}; "
        f"teste {split.test_start} a {split.test_end}; "
        f"{len(series_history)} series"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results and decision rule
# MAGIC
# MAGIC The candidate can move toward production only if it beats zero on
# MAGIC product-location WAPE in **every** fold. Even then, review the totals and
# MAGIC bias table before replacing the job implementation. The goal is to correct
# MAGIC an implausible portfolio forecast while preserving the product-by-store
# MAGIC delivery requirement.

# COMMAND ----------

results_df = spark.createDataFrame(metric_rows)  # noqa: F821
show_result(results_df.orderBy("fold_index", "method"))

comparison_rows = []
for method in LOCATION_ALLOCATION_METHODS:
    candidate_rows = [row for row in metric_rows if row["method"] == method]
    zero_by_fold = {
        row["fold_index"]: row for row in metric_rows if row["method"] == "zero"
    }
    comparison_rows.append(
        {
            "method": method,
            "beats_zero_at_sku_location_in_all_folds": all(
                row["sku_location_wape"]
                < zero_by_fold[row["fold_index"]]["sku_location_wape"]
                for row in candidate_rows
            ),
            "beats_zero_at_location_in_all_folds": all(
                row["location_wape"] < zero_by_fold[row["fold_index"]]["location_wape"]
                for row in candidate_rows
            ),
            "status": "research_only_review_metrics_before_promotion",
        }
    )
comparison_df = spark.createDataFrame(comparison_rows)  # noqa: F821
show_result(comparison_df.orderBy("method"))

exit_payload = {
    "experiment": "location_to_product_location_two_layer_v1",
    "folds": len(FOLD_CUTOFFS),
    "metrics_by_fold": metric_rows,
    "methods": comparison_rows,
    "status": "research_only_no_forecast_published",
}
print(json.dumps(exit_payload, sort_keys=True))
try:
    dbutils.notebook.exit(json.dumps(exit_payload, sort_keys=True))  # noqa: F821
except NameError:
    pass
