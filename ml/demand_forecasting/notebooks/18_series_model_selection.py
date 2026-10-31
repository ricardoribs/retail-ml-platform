# Databricks notebook source
# ruff: noqa: E402

# COMMAND ----------

"""Selects the winning product-location baseline across three backtests.

This step bridges evaluation and delivery: instead of approving a strategy based on
a segment-wide average, each series receives the model that consistently beat the
zero baseline in three time windows. Selection uses only closed months before the
forecast."""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

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

from ml.demand_forecasting.src.baseline import recursive_baseline_predictions
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
from ml.demand_forecasting.src.series_model_selection import (
    MODEL_NAMES,
    select_series_champion,
)

# COMMAND ----------


def show_result(df: DataFrame, rows: int = 100) -> None:
    """Displays a DataFrame in Databricks, with a local fallback."""
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(rows, truncate=False)


def month_end(month_start):
    return shift_months(month_start, 1) - timedelta(days=1)


FEATURE_MATRIX_TABLE = fq_table(INTERMEDIATE_SCHEMA, "int_forecast_feature_matrix")
METRICS_TABLE = fq_table(MARTS_SCHEMA, "fct_forecast_series_backtest_metrics")
SELECTION_TABLE = fq_table(MARTS_SCHEMA, "fct_forecast_series_model_selection")
SELECTION_RUN_ID = f"series_selection_{FORECAST_CONFIG.snapshot_id}"
MODEL_VERSION = "series_baseline_champion_v2_sparse_zero"
TEST_WINDOW_MONTHS = len(FORECAST_CONFIG.horizon_months)
FOLD_OFFSETS = (0, TEST_WINDOW_MONTHS, TEST_WINDOW_MONTHS * 2)
FOLD_CUTOFFS = tuple(
    month_end(shift_months(FORECAST_CONFIG.cutoff_date, -offset))
    for offset in FOLD_OFFSETS
)

print("Granular selection parameters")
print(f"  Features       : {FEATURE_MATRIX_TABLE}")
print(f"  Metrics        : {METRICS_TABLE}")
print(f"  Selection      : {SELECTION_TABLE}")
print(f"  Run ID         : {SELECTION_RUN_ID}")
print(f"  Cortes         : {', '.join(str(cutoff) for cutoff in FOLD_CUTOFFS)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Leakage-free recursive backtests
# MAGIC
# MAGIC Each fold uses only history from before its evaluation window. The forecast
# MAGIC for the next month uses the previous month's forecast, never the
# MAGIC actuals from the horizon.

# COMMAND ----------

feature_matrix_df = spark.table(FEATURE_MATRIX_TABLE)  # noqa: F821
metric_records = []
created_at = datetime.now(timezone.utc).replace(tzinfo=None)

for fold_index, fold_cutoff in enumerate(FOLD_CUTOFFS, start=1):
    split = build_temporal_split(fold_cutoff, TEST_WINDOW_MONTHS)
    target_months = tuple(
        shift_months(split.test_start, offset) for offset in range(TEST_WINDOW_MONTHS)
    )
    test_df = feature_matrix_df.filter(
        (F.col("demand_month") >= F.lit(split.test_start))
        & (F.col("demand_month") <= F.lit(split.test_end))
    )
    actual_rows = test_df.select(
        "demand_month", *SERIES_KEY_COLUMNS, LABEL_COLUMN
    ).collect()
    if not actual_rows:
        raise ValueError(f"Fold {fold_index} has no actuals to evaluate")
    history_states = collect_backtest_history(
        feature_matrix_df, test_df, split.train_end
    )
    predictions_by_model = recursive_baseline_predictions(history_states, target_months)
    actual_by_key = {
        (row["demand_month"], row["product_variant_id"], row["location_id"]): float(
            row[LABEL_COLUMN] or 0.0
        )
        for row in actual_rows
    }
    if len(actual_by_key) != len(actual_rows):
        raise ValueError("Duplicate actuals were found in the granular backtest")

    for model_name in MODEL_NAMES:
        prediction_rows = predictions_by_model[model_name]
        prediction_by_key = {
            (
                row["demand_month"],
                row["product_variant_id"],
                row["location_id"],
            ): float(row["prediction"])
            for row in prediction_rows
        }
        if set(prediction_by_key) != set(actual_by_key):
            raise AssertionError(
                f"{model_name} did not exactly cover the actuals for fold {fold_index}"
            )
        series_values: dict[tuple[int, int], list[tuple[float, float]]] = defaultdict(
            list
        )
        for (
            demand_month,
            product_variant_id,
            location_id,
        ), actual in actual_by_key.items():
            series_values[(product_variant_id, location_id)].append(
                (
                    actual,
                    prediction_by_key[(demand_month, product_variant_id, location_id)],
                )
            )
        for (product_variant_id, location_id), values in series_values.items():
            actual_abs_sum = sum(abs(actual) for actual, _ in values)
            absolute_error_sum = sum(
                abs(actual - predicted) for actual, predicted in values
            )
            metric_records.append(
                {
                    "run_id": SELECTION_RUN_ID,
                    "product_variant_id": product_variant_id,
                    "location_id": location_id,
                    "model_name": model_name,
                    "model_version": MODEL_VERSION,
                    "fold_index": fold_index,
                    "train_end_month": split.train_end,
                    "test_start_month": split.test_start,
                    "test_end_month": split.test_end,
                    "actual_abs_sum": float(actual_abs_sum),
                    "absolute_error_sum": float(absolute_error_sum),
                    "positive_actual_rows": sum(
                        1 for actual, _ in values if actual > 0.0
                    ),
                    "evaluated_rows": len(values),
                    "wape": (
                        float(absolute_error_sum / actual_abs_sum * 100.0)
                        if actual_abs_sum > 0.0
                        else None
                    ),
                    "created_at": created_at,
                }
            )

if not metric_records:
    raise ValueError("The granular selection produced no metrics")
print(f"Backtest metrics generated: {len(metric_records)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Per-series champion
# MAGIC
# MAGIC A candidate is promoted only if it beats zero in two of the three windows,
# MAGIC reduces combined WAPE by at least two percentage points and has at least
# MAGIC two months with observed demand in validation. If there are zero or
# MAGIC one intermittent occurrence. The zero baseline remains published: it is the best model
# MAGIC observed-demand months, the policy records a monthly review due to low confidence.

# COMMAND ----------

metrics_by_series: dict[tuple[int, int], list[dict]] = defaultdict(list)
for metric in metric_records:
    metrics_by_series[(metric["product_variant_id"], metric["location_id"])].append(
        metric
    )

selection_records = []
for (product_variant_id, location_id), series_metrics in metrics_by_series.items():
    decision = select_series_champion(series_metrics)
    selection_records.append(
        {
            "run_id": SELECTION_RUN_ID,
            "product_variant_id": product_variant_id,
            "location_id": location_id,
            "selected_model_name": decision.selected_model_name,
            "model_version": MODEL_VERSION,
            "validation_status": decision.validation_status,
            "combined_wape": decision.combined_wape,
            "zero_wape": decision.zero_wape,
            "wape_improvement": decision.wape_improvement,
            "validation_folds": decision.validation_folds,
            "folds_won_against_zero": decision.folds_won_against_zero,
            "positive_actual_rows": decision.positive_actual_rows,
            "selection_reason": decision.selection_reason,
            "created_at": created_at,
        }
    )

if len(selection_records) != len(metrics_by_series):
    raise AssertionError("The selection did not cover all evaluated series")

metrics_df = spark.createDataFrame(metric_records)  # noqa: F821
selection_df = spark.createDataFrame(selection_records)  # noqa: F821
show_result(
    selection_df.groupBy("selected_model_name", "validation_status")
    .count()
    .orderBy("selected_model_name", "validation_status")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Idempotent persistence and inference contract

# COMMAND ----------

for table_name, source_df, merge_condition in (
    (
        METRICS_TABLE,
        metrics_df,
        "target.run_id = source.run_id "
        "AND target.product_variant_id = source.product_variant_id "
        "AND target.location_id = source.location_id "
        "AND target.model_name = source.model_name "
        "AND target.fold_index = source.fold_index",
    ),
    (
        SELECTION_TABLE,
        selection_df,
        "target.run_id = source.run_id "
        "AND target.product_variant_id = source.product_variant_id "
        "AND target.location_id = source.location_id",
    ),
):
    if spark.catalog.tableExists(table_name):  # noqa: F821
        from delta.tables import DeltaTable

        target = DeltaTable.forName(spark, table_name)  # noqa: F821
        (
            target.alias("target")
            .merge(source_df.alias("source"), merge_condition)
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .whenNotMatchedBySourceDelete(
                condition=f"target.run_id = '{SELECTION_RUN_ID}'"
            )
            .execute()
        )
    else:
        source_df.write.format("delta").mode("append").saveAsTable(table_name)

persisted_selection_count = (
    spark.table(SELECTION_TABLE)  # noqa: F821
    .filter(F.col("run_id") == F.lit(SELECTION_RUN_ID))
    .count()
)
if persisted_selection_count != len(selection_records):
    raise AssertionError("Per-series selection persistence is incomplete")

set_task_value("series_selection_run_id", SELECTION_RUN_ID)
set_task_value("series_selection_refreshed", True)
print(f"Selection persisted to: {SELECTION_TABLE}")
print("GRANULAR MODEL SELECTION COMPLETE")
