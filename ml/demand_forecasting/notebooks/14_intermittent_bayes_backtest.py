# Databricks notebook source
# ruff: noqa: E402

"""Read-only backtest of candidates for intermittent monthly sales.

Rules use only sales before the first month of each holdout. Compares a probabilistic
mixture with fixed Croston/SBA/TSB candidates. No candidate is written, promoted,
or connected to the operational job."""

from __future__ import annotations

import json
import math
import sys
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
from ml.demand_forecasting.src.hierarchical import evaluate_forecasts
from ml.demand_forecasting.src.intermittent import (
    INTERMITTENT_METHODS,
    build_intermittent_forecasts,
)
from ml.demand_forecasting.src.intermittent_bayes import (
    BASELINE_METHODS,
    HURDLE_METHODS,
    build_hurdle_forecasts,
)
from ml.demand_forecasting.src.metrics import calculate_metrics
from ml.demand_forecasting.src.modeling import (
    LABEL_COLUMN,
    SERIES_KEY_COLUMNS,
    build_temporal_split,
    prepare_temporal_frames,
    shift_months,
)
from ml.demand_forecasting.src.selective_intermittent import (
    OCCURRENCE_F_BETA,
    PROBABILITY_THRESHOLDS,
    SELECTIVE_MODEL_NAME,
    build_conditional_quantity_forecast,
    build_occurrence_probabilities,
    build_selective_candidates,
    select_threshold_from_history,
)
from ml.demand_forecasting.src.two_stage import calculate_occurrence_metrics


def show_result(df: DataFrame) -> None:
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(200, truncate=False)


def finite_round(value: float, digits: int = 2) -> float | None:
    return round(value, digits) if math.isfinite(value) else None


def groups_as_of(
    history_df: DataFrame,
    origin: date,
    expected_series: set[tuple[int, int]],
) -> dict[tuple[int, int], tuple[int, str]]:
    """Get the category known at the internal origin date."""
    latest = Window.partitionBy(*SERIES_KEY_COLUMNS).orderBy(
        F.col("demand_month").desc()
    )
    rows = (
        history_df.filter(F.col("demand_month") < F.lit(origin))
        .withColumn("_rank", F.row_number().over(latest))
        .filter(F.col("_rank") == 1)
        .select(
            *SERIES_KEY_COLUMNS,
            F.coalesce(F.col("category_name"), F.lit("__missing__")).alias("_category"),
        )
        .collect()
    )
    groups = {
        (row["product_variant_id"], row["location_id"]): (
            row["location_id"],
            row["_category"],
        )
        for row in rows
        if (row["product_variant_id"], row["location_id"]) in expected_series
    }
    if set(groups) != expected_series:
        raise ValueError(f"Incomplete metadata at internal origin {origin}")
    return groups


def actual_for_targets(
    history_df: DataFrame,
    target_months: tuple[date, ...],
    expected_series: set[tuple[int, int]],
) -> dict[tuple[date, int, int], float]:
    """Read actuals only to evaluate an internal window that has already closed."""
    rows = [
        row
        for row in (
            history_df.filter(F.col("demand_month").isin(*target_months))
            .select("demand_month", *SERIES_KEY_COLUMNS, LABEL_COLUMN)
            .collect()
        )
        if (row["product_variant_id"], row["location_id"]) in expected_series
    ]
    actual = {
        (row["demand_month"], row["product_variant_id"], row["location_id"]): float(
            row[LABEL_COLUMN]
        )
        for row in rows
    }
    expected_keys = {
        (month, product_id, location_id)
        for month in target_months
        for product_id, location_id in expected_series
    }
    if len(actual) != len(rows) or set(actual) != expected_keys:
        raise ValueError("The inner window contains duplicate or incomplete keys")
    return actual


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Cutoffs and cohorts known before the test
# MAGIC
# MAGIC The target is **observed paid sales**, not estimated lost demand. Each
# MAGIC `cutoff_date` is the last day of the three-month evaluation window: the
# MAGIC cutoffs below test September–November 2025 and June–August 2026. Forecast
# MAGIC history ends before the first month in each window.

# COMMAND ----------

dbutils.widgets.text("cutoff_dates", "2025-11-30,2026-08-31")  # noqa: F821
cutoff_dates = tuple(
    date.fromisoformat(value.strip())
    for value in dbutils.widgets.get("cutoff_dates").split(",")  # noqa: F821
    if value.strip()
)
if len(cutoff_dates) != 2 or len(set(cutoff_dates)) != 2:
    raise ValueError("Informe dois cortes distintos no formato YYYY-MM-DD")
for cutoff in cutoff_dates:
    if cutoff != shift_months(cutoff, 1) - timedelta(days=1):
        raise ValueError(f"The cutoff must be the last day of the month: {cutoff}")
    if cutoff > FORECAST_CONFIG.cutoff_date:
        raise ValueError(f"Cutoff is after the last closed month: {cutoff}")

feature_matrix_df = spark.table(  # noqa: F821
    fq_table(INTERMEDIATE_SCHEMA, "int_forecast_feature_matrix")
).filter(F.col("demand_month") <= F.lit(max(cutoff_dates)))
duplicate_keys = (
    feature_matrix_df.groupBy("demand_month", *SERIES_KEY_COLUMNS)
    .count()
    .filter(F.col("count") != 1)
    .count()
)
invalid_labels = feature_matrix_df.filter(
    F.col(LABEL_COLUMN).isNull()
    | F.isnan(F.col(LABEL_COLUMN).cast("double"))
    | (F.col(LABEL_COLUMN) < 0)
).count()
if duplicate_keys or invalid_labels:
    raise ValueError(
        f"Invalid matrix: {duplicate_keys} duplicate keys, "
        f"{invalid_labels} invalid quantities"
    )

score_rows: list[dict[str, object]] = []
monthly_rows: list[dict[str, object]] = []
coverage_rows: list[dict[str, object]] = []
calibration_rows: list[dict[str, object]] = []
occurrence_rows: list[dict[str, object]] = []
magnitude_rows: list[dict[str, object]] = []

for cutoff in cutoff_dates:
    split = build_temporal_split(cutoff, test_months=3)
    target_months = tuple(shift_months(split.test_start, i) for i in range(3))
    eligible_df, _, test_df = prepare_temporal_frames(
        feature_matrix_df,
        split,
        FORECAST_CONFIG.min_history_months,
        FORECAST_CONFIG.min_positive_demand_months,
    )
    eligible_count = eligible_df.count()
    if not eligible_count:
        raise ValueError(f"No eligible series at cutoff {cutoff}")

    history_df = feature_matrix_df.filter(
        (F.col("demand_month") >= F.lit(shift_months(split.test_start, -24)))
        & (F.col("demand_month") <= F.lit(split.train_end))
    ).join(eligible_df.select(*SERIES_KEY_COLUMNS), list(SERIES_KEY_COLUMNS))
    latest = Window.partitionBy(*SERIES_KEY_COLUMNS).orderBy(
        F.col("demand_month").desc()
    )
    metadata_df = (
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
    metadata_rows = metadata_df.collect()
    series_groups = {
        (row["product_variant_id"], row["location_id"]): (
            row["location_id"],
            row["_category"],
        )
        for row in metadata_rows
    }
    series_types = {
        (row["product_variant_id"], row["location_id"]): row["_location_type"]
        for row in metadata_rows
    }
    if len(metadata_rows) != eligible_count or len(series_groups) != eligible_count:
        raise ValueError(f"Metadata is incomplete or duplicated at {cutoff}")
    store_series = {
        series
        for series, location_type in series_types.items()
        if location_type == "store"
    }
    if not store_series:
        raise ValueError(f"No eligible stores at {cutoff}")

    series_history = {series: {} for series in series_groups}
    for row in history_df.select(
        "demand_month", *SERIES_KEY_COLUMNS, LABEL_COLUMN
    ).collect():
        series = (row["product_variant_id"], row["location_id"])
        month = row["demand_month"]
        if month in series_history[series]:
            raise ValueError(f"Duplicate history for {cutoff}: {series}, {month}")
        series_history[series][month] = float(row[LABEL_COLUMN])
    store_groups = {series: series_groups[series] for series in store_series}
    store_history = {series: series_history[series] for series in store_series}

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
        raise ValueError(f"Incomplete or duplicate holdout for {cutoff}")

    full_test_df = feature_matrix_df.filter(
        (F.col("demand_month") >= F.lit(split.test_start))
        & (F.col("demand_month") <= F.lit(split.test_end))
    )
    full_summary = full_test_df.agg(
        F.countDistinct(*SERIES_KEY_COLUMNS).alias("all_series"),
        F.sum(LABEL_COLUMN).alias("all_qty"),
    ).first()
    if not full_summary["all_series"] or not full_summary["all_qty"]:
        raise ValueError(f"Holdout has no total volume at {cutoff}")
    coverage_rows.append(
        {
            "cutoff": cutoff.isoformat(),
            "eligible_series": eligible_count,
            "all_grid_series": int(full_summary["all_series"]),
            "eligible_series_pct": round(
                100 * eligible_count / int(full_summary["all_series"]), 2
            ),
            "eligible_actual_qty_pct": round(
                100 * sum(actual.values()) / float(full_summary["all_qty"]), 2
            ),
            "eligible_zero_sale_rows_pct": round(
                100 * sum(value == 0 for value in actual.values()) / len(actual), 2
            ),
        }
    )

    inner_origins = tuple(
        shift_months(split.test_start, offset) for offset in (-9, -6, -3)
    )
    calibration_actual: dict[tuple[date, int, int], float] = {}
    calibration_probabilities: dict[tuple[date, int, int], float] = {}
    calibration_candidates = {threshold: {} for threshold in PROBABILITY_THRESHOLDS}
    for inner_origin in inner_origins:
        inner_targets = tuple(shift_months(inner_origin, offset) for offset in range(3))
        inner_history = {
            series: {
                month: quantity
                for month, quantity in history.items()
                if month < inner_origin
            }
            for series, history in store_history.items()
        }
        inner_groups = groups_as_of(history_df, inner_origin, store_series)
        inner_actual = actual_for_targets(
            history_df,
            inner_targets,
            store_series,
        )
        inner_candidates = build_selective_candidates(
            series_groups=inner_groups,
            series_history=inner_history,
            target_months=inner_targets,
            thresholds=PROBABILITY_THRESHOLDS,
        )
        inner_probabilities = build_occurrence_probabilities(
            series_groups=inner_groups,
            series_history=inner_history,
            target_months=inner_targets,
        )
        if set(inner_actual).intersection(calibration_actual):
            raise ValueError("Inner calibration windows overlap")
        if set(inner_probabilities) != set(inner_actual):
            raise ValueError("Probabilities do not cover the inner window")
        calibration_actual.update(inner_actual)
        calibration_probabilities.update(inner_probabilities)
        for threshold, predicted in inner_candidates.items():
            if set(predicted) != set(inner_actual):
                raise ValueError("The selective candidate does not cover the inner window")
            calibration_candidates[threshold].update(predicted)

    selection = select_threshold_from_history(
        actual=calibration_actual,
        probabilities=calibration_probabilities,
        thresholds=PROBABILITY_THRESHOLDS,
        beta=OCCURRENCE_F_BETA,
    )
    selected_inner_predictions = calibration_candidates[selection.threshold]
    ordered_inner_keys = tuple(sorted(calibration_actual))
    inner_forecast_metrics = calculate_metrics(
        [calibration_actual[key] for key in ordered_inner_keys],
        [selected_inner_predictions[key] for key in ordered_inner_keys],
    )
    inner_event_rate = sum(
        calibration_actual[key] > 0 for key in ordered_inner_keys
    ) / len(ordered_inner_keys)
    first_inner_training_months = tuple(
        shift_months(inner_origins[0], -offset) for offset in range(1, 13)
    )
    inner_training_events = [
        quantity > 0
        for history in store_history.values()
        for month, quantity in history.items()
        if month in first_inner_training_months
    ]
    inner_reference_probability = (
        sum(inner_training_events) / len(inner_training_events)
        if inner_training_events
        else 0.0
    )
    inner_occurrence_metrics = calculate_occurrence_metrics(
        [calibration_actual[key] for key in ordered_inner_keys],
        [calibration_probabilities[key] for key in ordered_inner_keys],
        inner_reference_probability,
        decision_threshold=selection.threshold,
    )
    selective_store_candidates = build_selective_candidates(
        series_groups=store_groups,
        series_history=store_history,
        target_months=target_months,
        thresholds=(selection.threshold,),
    )
    calibration_rows.append(
        {
            "cutoff": cutoff.isoformat(),
            "scope": "store_all",
            "series": len(store_series),
            "inner_origins": ",".join(origin.isoformat() for origin in inner_origins),
            "selected_probability_threshold": selection.threshold,
            "selection_metric": f"f{OCCURRENCE_F_BETA:g}",
            "inner_precision": finite_round(selection.precision, 4),
            "inner_recall": finite_round(selection.recall, 4),
            "inner_fbeta": finite_round(selection.fbeta, 4),
            "inner_average_precision": finite_round(
                inner_occurrence_metrics["average_precision"], 4
            ),
            "inner_roc_auc": finite_round(inner_occurrence_metrics["roc_auc"], 4),
            "inner_brier_skill_score": finite_round(
                inner_occurrence_metrics["brier_skill_score"], 4
            ),
            "inner_event_rate": finite_round(inner_event_rate, 4),
            "inner_wape_diagnostic": finite_round(
                float(inner_forecast_metrics["wape"])
            ),
            "inner_mae_diagnostic": finite_round(
                float(inner_forecast_metrics["mae"]), 3
            ),
            "inner_positive_forecast_rows": selection.predicted_positive_rows,
            "inner_true_positive_rows": selection.true_positive_rows,
            "inner_false_positive_rows": selection.false_positive_rows,
            "inner_false_negative_rows": selection.false_negative_rows,
            "detected_any_inner_sale": selection.has_detected_sales,
        }
    )

    forecasts = build_hurdle_forecasts(
        series_groups=series_groups,
        series_history=series_history,
        target_months=target_months,
    )
    forecasts.update(
        build_intermittent_forecasts(
            series_history=series_history,
            target_months=target_months,
        )
    )
    selective_forecast = dict(forecasts["zero"])
    selective_forecast.update(selective_store_candidates[selection.threshold])
    forecasts[SELECTIVE_MODEL_NAME] = selective_forecast
    outer_probabilities = build_occurrence_probabilities(
        series_groups=store_groups,
        series_history=store_history,
        target_months=target_months,
    )
    store_actual = {
        key: value for key, value in actual.items() if key[1:] in store_series
    }
    if set(outer_probabilities) != set(store_actual):
        raise ValueError(f"Store probabilities are incomplete at {cutoff}")
    outer_selection_score = select_threshold_from_history(
        actual=store_actual,
        probabilities=outer_probabilities,
        thresholds=(selection.threshold,),
        beta=OCCURRENCE_F_BETA,
    )
    last12 = tuple(shift_months(split.test_start, -offset) for offset in range(1, 13))
    training_events = [
        quantity > 0
        for history in store_history.values()
        for month, quantity in history.items()
        if month in last12
    ]
    reference_probability = (
        sum(training_events) / len(training_events) if training_events else 0.0
    )
    ordered_outer_keys = tuple(sorted(store_actual))
    outer_occurrence_metrics = calculate_occurrence_metrics(
        [store_actual[key] for key in ordered_outer_keys],
        [outer_probabilities[key] for key in ordered_outer_keys],
        reference_probability,
        decision_threshold=selection.threshold,
    )
    occurrence_rows.append(
        {
            "cutoff": cutoff.isoformat(),
            "scope": "store_all",
            "series": len(store_series),
            "selected_probability_threshold": selection.threshold,
            "selection_metric": f"f{OCCURRENCE_F_BETA:g}",
            "average_precision": finite_round(
                outer_occurrence_metrics["average_precision"], 4
            ),
            "roc_auc": finite_round(outer_occurrence_metrics["roc_auc"], 4),
            "brier_score": finite_round(outer_occurrence_metrics["brier_score"], 4),
            "brier_skill_score": finite_round(
                outer_occurrence_metrics["brier_skill_score"], 4
            ),
            "precision": finite_round(outer_selection_score.precision, 4),
            "recall": finite_round(outer_selection_score.recall, 4),
            "fbeta": finite_round(outer_selection_score.fbeta, 4),
            "true_positive_rows": outer_selection_score.true_positive_rows,
            "false_positive_rows": outer_selection_score.false_positive_rows,
            "false_negative_rows": outer_selection_score.false_negative_rows,
        }
    )
    conditional_quantity = build_conditional_quantity_forecast(
        series_groups=store_groups,
        series_history=store_history,
        target_months=target_months,
        probabilities=outer_probabilities,
    )
    positive_outer_keys = tuple(
        key for key in ordered_outer_keys if store_actual[key] > 0
    )
    conditional_metrics = calculate_metrics(
        [store_actual[key] for key in positive_outer_keys],
        [conditional_quantity[key] for key in positive_outer_keys],
    )
    magnitude_rows.append(
        {
            "cutoff": cutoff.isoformat(),
            "scope": "store_all",
            "series": len(store_series),
            "positive_rows": len(positive_outer_keys),
            "conditional_quantity_wape": finite_round(
                float(conditional_metrics["wape"])
            ),
            "conditional_quantity_mae": finite_round(
                float(conditional_metrics["mae"]), 3
            ),
        }
    )
    scopes = {"all": set(series_groups), "store_all": store_series}

    for method, predicted in forecasts.items():
        if set(predicted) != expected_keys:
            raise ValueError(f"Incomplete forecast: {method}, {cutoff}")
        for scope, selected_series in scopes.items():
            selected_actual = {
                key: qty for key, qty in actual.items() if key[1:] in selected_series
            }
            selected_predicted = {key: predicted[key] for key in selected_actual}
            selected_groups = {
                series: series_groups[series] for series in selected_series
            }
            metrics = evaluate_forecasts(
                selected_actual, selected_predicted, selected_groups
            )
            actual_qty = sum(selected_actual.values())
            predicted_qty = sum(selected_predicted.values())
            score_rows.append(
                {
                    "cutoff": cutoff.isoformat(),
                    "scope": scope,
                    "method": method,
                    "series": len(selected_series),
                    "actual_qty": round(actual_qty, 2),
                    "predicted_qty": round(predicted_qty, 2),
                    "bias_pct": finite_round(
                        100 * (predicted_qty - actual_qty) / actual_qty
                        if actual_qty
                        else math.nan
                    ),
                    "positive_forecast_pct": round(
                        100
                        * sum(value > 0 for value in selected_predicted.values())
                        / len(selected_predicted),
                        2,
                    ),
                    "false_positive_rows": sum(
                        selected_actual[key] == 0 and value > 0
                        for key, value in selected_predicted.items()
                    ),
                    "missed_sale_rows": sum(
                        selected_actual[key] > 0 and value == 0
                        for key, value in selected_predicted.items()
                    ),
                    "sku_location_wape": finite_round(metrics["sku_location"]["wape"]),
                    "sku_location_mae": finite_round(metrics["sku_location"]["mae"], 3),
                    "category_location_wape": finite_round(
                        metrics["category_location"]["wape"]
                    ),
                    "location_wape": finite_round(metrics["location"]["wape"]),
                }
            )
            for month in target_months:
                month_actual = {
                    key: qty for key, qty in selected_actual.items() if key[0] == month
                }
                month_predicted = {key: predicted[key] for key in month_actual}
                month_metrics = evaluate_forecasts(
                    month_actual, month_predicted, selected_groups
                )["sku_location"]
                monthly_rows.append(
                    {
                        "cutoff": cutoff.isoformat(),
                        "scope": scope,
                        "month": month.isoformat(),
                        "method": method,
                        "actual_qty": round(sum(month_actual.values()), 2),
                        "predicted_qty": round(sum(month_predicted.values()), 2),
                        "sku_location_wape": finite_round(month_metrics["wape"]),
                    }
                )
    print(
        f"{cutoff}: {eligible_count} eligible series, "
        f"{len(store_series)} in stores; "
        "no holdout sales were included in the forecasts"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Results and exploratory floor
# MAGIC
# MAGIC First compare `store_all` at the `sku_location_wape` grain, then
# MAGIC coverage, each month, volume bias, and months with no sales. Beating the model
# MAGIC zero at the aggregate level alone is not enough. This threshold does not approve production use.

# COMMAND ----------

show_result(spark.createDataFrame(coverage_rows).orderBy("cutoff"))  # noqa: F821
show_result(spark.createDataFrame(calibration_rows).orderBy("cutoff"))  # noqa: F821
show_result(spark.createDataFrame(occurrence_rows).orderBy("cutoff"))  # noqa: F821
show_result(spark.createDataFrame(magnitude_rows).orderBy("cutoff"))  # noqa: F821
show_result(  # noqa: F821
    spark.createDataFrame(score_rows).orderBy("cutoff", "scope", "method")  # noqa: F821
)
show_result(  # noqa: F821
    spark.createDataFrame(monthly_rows).orderBy(  # noqa: F821
        "cutoff", "scope", "month", "method"
    )
)

print("Exploratory floor at the product/store grain:")
approval_rows = []
for method in (*HURDLE_METHODS, *INTERMITTENT_METHODS, SELECTIVE_MODEL_NAME):
    passed_cuts = []
    for cutoff in cutoff_dates:
        cutoff_name = cutoff.isoformat()
        store_scores = {
            row["method"]: row
            for row in score_rows
            if row["cutoff"] == cutoff_name and row["scope"] == "store_all"
        }
        candidate = store_scores[method]
        references = [
            store_scores[name]["sku_location_wape"] for name in BASELINE_METHODS
        ]
        monthly_candidate = [
            row["sku_location_wape"]
            for row in monthly_rows
            if row["cutoff"] == cutoff_name
            and row["scope"] == "store_all"
            and row["method"] == method
        ]
        monthly_references = {
            month.isoformat(): min(
                row["sku_location_wape"]
                for row in monthly_rows
                if row["cutoff"] == cutoff_name
                and row["scope"] == "store_all"
                and row["month"] == month.isoformat()
                and row["method"] in BASELINE_METHODS
                and row["sku_location_wape"] is not None
            )
            for month in tuple(
                shift_months(build_temporal_split(cutoff, 3).test_start, i)
                for i in range(3)
            )
        }
        monthly_by_month = {
            row["month"]: row["sku_location_wape"]
            for row in monthly_rows
            if row["cutoff"] == cutoff_name
            and row["scope"] == "store_all"
            and row["method"] == method
        }
        passed = (
            candidate["sku_location_wape"] is not None
            and all(value is not None for value in references)
            and candidate["sku_location_wape"] < min(references)
            and candidate["positive_forecast_pct"] > 0
            and len(monthly_candidate) == 3
            and all(
                monthly_by_month.get(month) is not None
                and monthly_by_month[month] < baseline
                for month, baseline in monthly_references.items()
            )
        )
        passed_cuts.append(passed)
        approval_rows.append(
            {
                "method": method,
                "cutoff": cutoff_name,
                "sku_location_wape": candidate["sku_location_wape"],
                "best_baseline_wape": min(references)
                if None not in references
                else None,
                "positive_forecast_pct": candidate["positive_forecast_pct"],
                "passed_exploratory_gate": passed,
            }
        )
        print(
            f"  {method}, {cutoff_name}: WAPE={candidate['sku_location_wape']}, "
            f"melhor baseline={min(references) if None not in references else None}, "
            f"positive forecasts={candidate['positive_forecast_pct']}%, "
            f"passou piso={passed}"
        )
    print(f"  {method}: passou os dois cortes={all(passed_cuts)}")

print(
    "Beating the floor is not production approval. The target is observed sales, "
    "does not detect lost demand; also check coverage, bias, and the usefulness of "
    "positive forecasts before changing the job."
)

try:
    dbutils.notebook.exit(  # noqa: F821
        json.dumps(
            {
                "approval": approval_rows,
                "nested_calibration": calibration_rows,
                "occurrence": occurrence_rows,
                "conditional_magnitude": magnitude_rows,
            }
        )
    )
except NameError:
    pass
