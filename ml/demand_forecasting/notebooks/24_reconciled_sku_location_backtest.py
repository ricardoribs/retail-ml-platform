# Databricks notebook source
# ruff: noqa: E402

"""Temporal backtest of SKU-location forecasts reconciled with IPF.

The experiment forecasts product and location margins using seasonality known at the
cutoff, smooths the SKU-location distribution by category, and reconciles the matrix.
It does not write tables, change the job, or promote models. Holdout actuals are read
only to calculate metrics after forecasting."""

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
from ml.demand_forecasting.src.metrics import calculate_metrics
from ml.demand_forecasting.src.modeling import (
    LABEL_COLUMN,
    SERIES_KEY_COLUMNS,
    build_temporal_split,
    prepare_temporal_frames,
    shift_months,
)
from ml.demand_forecasting.src.reconciled_allocation import (
    RECONCILED_ALLOCATION_METHOD,
    build_reconciled_allocation_forecasts,
)


def show_result(df: DataFrame) -> None:
    """Display results in a notebook or in a job's output."""
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(200, truncate=False)


def month_end(month_start):
    return shift_months(month_start, 1) - timedelta(days=1)


def product_month_metrics(actual, predicted) -> dict[str, float]:
    """Measure the product-month margin in the same SKU-location cohort."""
    actual_totals: dict[tuple, float] = defaultdict(float)
    predicted_totals: dict[tuple, float] = defaultdict(float)
    for (month, product_id, _location_id), value in actual.items():
        actual_totals[(month, product_id)] += value
        predicted_totals[(month, product_id)] += predicted[
            (month, product_id, _location_id)
        ]
    return calculate_metrics(
        [actual_totals[key] for key in sorted(actual_totals)],
        [predicted_totals[key] for key in sorted(actual_totals)],
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Protocolo as-of
# MAGIC
# MAGIC The three folds cover three-month windows immediately before the
# MAGIC operational cutoff. The distribution and both margins use only months
# MAGIC start of each holdout.

TEST_WINDOW_MONTHS = len(FORECAST_CONFIG.horizon_months)
FOLD_OFFSETS = (0, TEST_WINDOW_MONTHS, TEST_WINDOW_MONTHS * 2)
FOLD_CUTOFFS = tuple(
    month_end(shift_months(FORECAST_CONFIG.cutoff_date, -offset))
    for offset in FOLD_OFFSETS
)
FEATURE_MATRIX_TABLE = fq_table(INTERMEDIATE_SCHEMA, "int_forecast_feature_matrix")

print("SKU-location backtest: reconciled product by location")
print(f"  Source: {FEATURE_MATRIX_TABLE}")
print(f"  Cortes: {', '.join(str(cutoff) for cutoff in FOLD_CUTOFFS)}")
print("  Persistence: none; read-only experiment")

feature_matrix_df = spark.table(FEATURE_MATRIX_TABLE)  # noqa: F821
duplicate_count = (
    feature_matrix_df.groupBy("demand_month", *SERIES_KEY_COLUMNS)
    .count()
    .filter(F.col("count") != F.lit(1))
    .count()
)
invalid_demand = feature_matrix_df.filter(
    F.col(LABEL_COLUMN).isNull() | (F.col(LABEL_COLUMN) < F.lit(0))
).count()
if duplicate_count or invalid_demand:
    raise ValueError(
        f"Invalid matrix: {duplicate_count} duplicate keys and "
        f"{invalid_demand} demandas nulas ou negativas"
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Backtest: margins, shrinkage, and reconciliation

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

    category_df = history_df.select(
        *SERIES_KEY_COLUMNS,
        F.coalesce(F.col("category_name"), F.lit("__missing__")).alias("category_name"),
    ).distinct()
    ambiguous_category = (
        category_df.groupBy(*SERIES_KEY_COLUMNS)
        .count()
        .filter(F.col("count") != F.lit(1))
        .count()
    )
    if ambiguous_category:
        raise ValueError("Category differs within the same SKU-location series")

    series_history: dict[tuple[int, int], dict] = defaultdict(dict)
    for row in history_df.select(
        "demand_month", *SERIES_KEY_COLUMNS, LABEL_COLUMN
    ).collect():
        series = (row["product_variant_id"], row["location_id"])
        series_history[series][row["demand_month"]] = float(row[LABEL_COLUMN])
    series_groups = {
        (row["product_variant_id"], row["location_id"]): (
            row["location_id"],
            row["category_name"],
        )
        for row in category_df.collect()
    }
    expected_series = eligible_df.count()
    if len(series_history) != expected_series or set(series_history) != set(
        series_groups
    ):
        raise ValueError("History or categories do not cover the eligible cohort")

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

    forecasts = build_location_allocation_forecasts(
        series_history=series_history,
        target_months=target_months,
    )
    forecasts[RECONCILED_ALLOCATION_METHOD] = build_reconciled_allocation_forecasts(
        series_groups=series_groups,
        series_history=series_history,
        target_months=target_months,
    )
    forecasts["zero"] = dict.fromkeys(actual, 0.0)
    evaluation_groups = {
        series: (series[1], series_groups[series][1]) for series in series_groups
    }
    for method in ("zero", *LOCATION_ALLOCATION_METHODS, RECONCILED_ALLOCATION_METHOD):
        predicted = forecasts[method]
        metrics = evaluate_forecasts(actual, predicted, evaluation_groups)
        product_metrics = product_month_metrics(actual, predicted)
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
                "product_wape": round(product_metrics["wape"], 2),
                "location_wape": round(metrics["location"]["wape"], 2),
                "total_month_wape": round(metrics["total_month"]["wape"], 2),
            }
        )
    print(
        f"Fold {fold_index}: training through {split.train_end}; "
        f"teste {split.test_start} a {split.test_end}; {len(series_history)} series"
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Promotion criteria
# MAGIC
# MAGIC The reconciled forecast can be promoted only if it beats the zero benchmark on WAPE
# MAGIC SKU-location across all three folds without degrading the product and
# MAGIC location margins. A negative result is useful evidence; in that case, use
# MAGIC this method only as a distribution forecast for planning.

results_df = spark.createDataFrame(metric_rows)  # noqa: F821
show_result(results_df.orderBy("fold_index", "method"))

zero_by_fold = {
    row["fold_index"]: row for row in metric_rows if row["method"] == "zero"
}
candidate_rows = [
    row for row in metric_rows if row["method"] == RECONCILED_ALLOCATION_METHOD
]
approval = {
    "method": RECONCILED_ALLOCATION_METHOD,
    "beats_zero_sku_location_in_all_folds": all(
        row["sku_location_wape"] < zero_by_fold[row["fold_index"]]["sku_location_wape"]
        for row in candidate_rows
    ),
    "beats_zero_product_in_all_folds": all(
        row["product_wape"] < zero_by_fold[row["fold_index"]]["product_wape"]
        for row in candidate_rows
    ),
    "beats_zero_location_in_all_folds": all(
        row["location_wape"] < zero_by_fold[row["fold_index"]]["location_wape"]
        for row in candidate_rows
    ),
    "status": "research_only_no_forecast_published",
}
show_result(spark.createDataFrame([approval]))  # noqa: F821

exit_payload = {
    "experiment": "product_location_reconciled_ipf_v1",
    "folds": len(FOLD_CUTOFFS),
    "metrics_by_fold": metric_rows,
    "approval": approval,
}
print(json.dumps(exit_payload, sort_keys=True))
try:
    dbutils.notebook.exit(json.dumps(exit_payload, sort_keys=True))  # noqa: F821
except NameError:
    pass
