# Databricks notebook source
# ruff: noqa: E402, F821

"""Stratifies the SKU-store backtest by ADI/CV2 without publishing a forecast.

WAPE remains a required error metric, but it is not used to conclude that forecasting
zero is the best replenishment decision. For each Syntetos-Boylan profile, the notebook
shows quantity error, sales coverage, and false positives for the baselines and
Croston/SBA/TSB.

No table is written and no model is promoted. Test months are never used to calculate
ADI/CV2 or candidate forecasts."""

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
from ml.demand_forecasting.src.intermittent import (
    INTERMITTENT_METHODS,
    build_intermittent_forecasts,
    build_tsb_occurrence_probabilities,
)
from ml.demand_forecasting.src.intermittent_bayes import (
    BASELINE_METHODS,
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
from ml.demand_forecasting.src.segmentation import classify_intermittent_pattern
from ml.demand_forecasting.src.two_stage import calculate_occurrence_metrics


def show_result(df: DataFrame) -> None:
    """Displays results in Databricks with a local fallback."""
    try:
        display(df)
    except NameError:
        df.show(200, truncate=False)


def finite_round(value: float, digits: int = 2) -> float | None:
    """Convert undefined metrics, such as WAPE with no sales, to NULL."""
    return round(value, digits) if math.isfinite(value) else None


def evaluate_segment_method(
    *,
    actual: dict[tuple[date, int, int], float],
    predicted: dict[tuple[date, int, int], float],
) -> dict[str, float | int | None]:
    """Measure quantity and coverage without confusing a forecast with the final policy."""
    if set(actual) != set(predicted):
        raise ValueError("Forecast does not exactly cover the evaluated set")

    keys = tuple(sorted(actual))
    actual_values = [actual[key] for key in keys]
    predicted_values = [predicted[key] for key in keys]
    metrics = calculate_metrics(actual_values, predicted_values)
    true_positive_rows = sum(actual[key] > 0 and predicted[key] > 0 for key in keys)
    false_positive_rows = sum(actual[key] == 0 and predicted[key] > 0 for key in keys)
    false_negative_rows = sum(actual[key] > 0 and predicted[key] == 0 for key in keys)
    forecast_positive_rows = true_positive_rows + false_positive_rows
    actual_positive_rows = true_positive_rows + false_negative_rows

    return {
        "actual_qty": round(sum(actual_values), 2),
        "predicted_qty": round(sum(predicted_values), 2),
        "wape": finite_round(float(metrics["wape"])),
        "mae": finite_round(float(metrics["mae"]), 3),
        "forecast_positive_rows": forecast_positive_rows,
        "true_positive_rows": true_positive_rows,
        "false_positive_rows": false_positive_rows,
        "false_negative_rows": false_negative_rows,
        "precision": finite_round(
            true_positive_rows / forecast_positive_rows
            if forecast_positive_rows
            else 0.0,
            4,
        ),
        "recall": finite_round(
            true_positive_rows / actual_positive_rows
            if actual_positive_rows
            else math.nan,
            4,
        ),
    }


def tsb_precision_recall_curve(
    *,
    actual: dict[tuple[date, int, int], float],
    probabilities: dict[tuple[date, int, int], float],
) -> list[dict[str, float | int]]:
    """Measure all TSB operating points without selecting one for publication.

    Thresholds are the distinct probability values in the holdout itself.
    This describes the signal's ability to distinguish cases; any future choice
    A production threshold must be calibrated on earlier windows.
    """
    if set(actual) != set(probabilities):
        raise ValueError("TSB probabilities do not cover the same holdout")
    ranked_events = sorted(
        ((float(probabilities[key]), float(actual[key]) > 0) for key in actual),
        reverse=True,
    )
    if not ranked_events:
        raise ValueError("TSB curve has no probabilities")

    total_positive_rows = sum(event for _, event in ranked_events)
    true_positive_rows = 0
    forecast_positive_rows = 0
    rows = []
    cursor = 0
    while cursor < len(ranked_events):
        threshold = ranked_events[cursor][0]
        end = cursor + 1
        while end < len(ranked_events) and ranked_events[end][0] == threshold:
            end += 1
        event_group = ranked_events[cursor:end]
        forecast_positive_rows += len(event_group)
        true_positive_rows += sum(event for _, event in event_group)
        false_positive_rows = forecast_positive_rows - true_positive_rows
        false_negative_rows = total_positive_rows - true_positive_rows
        precision = true_positive_rows / forecast_positive_rows
        recall = (
            true_positive_rows / total_positive_rows if total_positive_rows else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        rows.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "forecast_positive_rows": forecast_positive_rows,
                "true_positive_rows": true_positive_rows,
                "false_positive_rows": false_positive_rows,
                "false_negative_rows": false_negative_rows,
            }
        )
        cursor = end
    return rows


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Temporal cutoffs and SKU-store cohort
# MAGIC
# MAGIC Each cutoff is the last day of a three-month holdout. ADI/CV² and all
# MAGIC forecasts use only history from before the first evaluated month.

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

profile_rows: list[dict[str, object]] = []
score_rows: list[dict[str, object]] = []
profile_by_cutoff: dict[date, dict[tuple[int, int], str]] = {}
tsb_pr_rows: list[dict[str, object]] = []

for cutoff in cutoff_dates:
    split = build_temporal_split(cutoff, test_months=3)
    target_months = tuple(shift_months(split.test_start, index) for index in range(3))
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
    metadata_rows = (
        history_df.withColumn("_rank", F.row_number().over(latest))
        .filter(F.col("_rank") == 1)
        .select(
            *SERIES_KEY_COLUMNS,
            F.coalesce(F.col("category_name"), F.lit("__missing__")).alias("_category"),
            F.coalesce(F.col("location_type"), F.lit("__missing__")).alias(
                "_location_type"
            ),
        )
        .collect()
    )
    series_groups = {
        (row["product_variant_id"], row["location_id"]): (
            row["location_id"],
            row["_category"],
        )
        for row in metadata_rows
    }
    store_series = {
        (row["product_variant_id"], row["location_id"])
        for row in metadata_rows
        if row["_location_type"] == "store"
    }
    if len(series_groups) != eligible_count or not store_series:
        raise ValueError(f"Series metadata is incomplete at cutoff {cutoff}")

    series_history = {series: {} for series in series_groups}
    for row in history_df.select(
        "demand_month", *SERIES_KEY_COLUMNS, LABEL_COLUMN
    ).collect():
        series = (row["product_variant_id"], row["location_id"])
        month = row["demand_month"]
        if month in series_history[series]:
            raise ValueError(f"Duplicate history: {series}, {month}")
        series_history[series][month] = float(row[LABEL_COLUMN])

    store_history = {series: series_history[series] for series in store_series}
    store_groups = {series: series_groups[series] for series in store_series}
    pattern_by_series = {}
    for series, history in store_history.items():
        profile = classify_intermittent_pattern(tuple(sorted(history.items())))
        pattern_by_series[series] = profile.demand_pattern
        profile_rows.append(
            {
                "cutoff": cutoff.isoformat(),
                "demand_pattern": profile.demand_pattern,
                "series": 1,
                "history_months": profile.history_month_count,
                "positive_history_months": profile.positive_demand_month_count,
                "adi": profile.adi,
                "cv_squared": profile.cv_squared,
            }
        )
    profile_by_cutoff[cutoff] = pattern_by_series

    actual = {
        (row["demand_month"], row["product_variant_id"], row["location_id"]): float(
            row[LABEL_COLUMN]
        )
        for row in test_df.select(
            "demand_month", *SERIES_KEY_COLUMNS, LABEL_COLUMN
        ).collect()
        if (row["product_variant_id"], row["location_id"]) in store_series
    }
    expected_keys = {
        (month, product_id, location_id)
        for month in target_months
        for product_id, location_id in store_series
    }
    if set(actual) != expected_keys:
        raise ValueError(f"Store holdout is incomplete at cutoff {cutoff}")

    forecasts = build_hurdle_forecasts(
        series_groups=store_groups,
        series_history=store_history,
        target_months=target_months,
    )
    forecasts.update(
        build_intermittent_forecasts(
            series_history=store_history,
            target_months=target_months,
        )
    )
    intermittent_series = {
        series
        for series, pattern in pattern_by_series.items()
        if pattern == "intermittent"
    }
    intermittent_actual = {
        key: quantity
        for key, quantity in actual.items()
        if key[1:] in intermittent_series
    }
    if intermittent_actual:
        tsb_probabilities = build_tsb_occurrence_probabilities(
            series_history=store_history,
            target_months=target_months,
        )
        intermittent_probabilities = {
            key: probability
            for key, probability in tsb_probabilities.items()
            if key[1:] in intermittent_series
        }
        curve_rows = tsb_precision_recall_curve(
            actual=intermittent_actual,
            probabilities=intermittent_probabilities,
        )
        historical_events = [
            quantity > 0
            for series in intermittent_series
            for quantity in store_history[series].values()
        ]
        reference_probability = sum(historical_events) / len(historical_events)
        occurrence_metrics = calculate_occurrence_metrics(
            [intermittent_actual[key] for key in sorted(intermittent_actual)],
            [intermittent_probabilities[key] for key in sorted(intermittent_actual)],
            reference_probability,
        )
        for row in curve_rows:
            tsb_pr_rows.append(
                {
                    "cutoff": cutoff.isoformat(),
                    "demand_pattern": "intermittent",
                    "roc_auc": finite_round(occurrence_metrics["roc_auc"], 4),
                    "average_precision": finite_round(
                        occurrence_metrics["average_precision"], 4
                    ),
                    **row,
                }
            )
    methods = (*BASELINE_METHODS, *INTERMITTENT_METHODS)
    patterns = sorted(set(pattern_by_series.values()))
    for pattern in patterns:
        selected_series = {
            series
            for series, series_pattern in pattern_by_series.items()
            if series_pattern == pattern
        }
        selected_actual = {
            key: quantity
            for key, quantity in actual.items()
            if key[1:] in selected_series
        }
        if not selected_actual:
            continue
        for method in methods:
            selected_predicted = {
                key: quantity
                for key, quantity in forecasts[method].items()
                if key[1:] in selected_series
            }
            metrics = evaluate_segment_method(
                actual=selected_actual,
                predicted=selected_predicted,
            )
            score_rows.append(
                {
                    "cutoff": cutoff.isoformat(),
                    "demand_pattern": pattern,
                    "method": method,
                    "series": len(selected_series),
                    **metrics,
                }
            )
    print(
        f"{cutoff}: {len(store_series)} SKU-store series; "
        f"perfis={{{', '.join(sorted(set(pattern_by_series.values())))}}}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Results by profile
# MAGIC
# MAGIC `precision` and `recall` here use only whether the forecast is
# MAGIC positive. These are coverage diagnostics, not an ordering policy.
# MAGIC The test does not observe inventory, cost, or lost demand, so it cannot
# MAGIC to establish the economic value of a replenishment policy.

# COMMAND ----------

profile_df = spark.createDataFrame(profile_rows)  # noqa: F821
score_df = spark.createDataFrame(score_rows)  # noqa: F821
profile_summary_rows = []
for cutoff_name, demand_pattern in sorted(
    {(row["cutoff"], row["demand_pattern"]) for row in profile_rows}
):
    profiles = [
        row
        for row in profile_rows
        if row["cutoff"] == cutoff_name and row["demand_pattern"] == demand_pattern
    ]
    adi_values = [float(row["adi"]) for row in profiles if row["adi"] is not None]
    cv_squared_values = [
        float(row["cv_squared"]) for row in profiles if row["cv_squared"] is not None
    ]
    profile_summary_rows.append(
        {
            "cutoff": cutoff_name,
            "demand_pattern": demand_pattern,
            "series": len(profiles),
            "avg_history_months": finite_round(
                sum(float(row["history_months"]) for row in profiles) / len(profiles)
            ),
            "avg_positive_history_months": finite_round(
                sum(float(row["positive_history_months"]) for row in profiles)
                / len(profiles)
            ),
            "avg_adi": finite_round(sum(adi_values) / len(adi_values))
            if adi_values
            else None,
            "avg_cv_squared": finite_round(
                sum(cv_squared_values) / len(cv_squared_values)
            )
            if cv_squared_values
            else None,
        }
    )
show_result(
    profile_df.groupBy("cutoff", "demand_pattern")
    .agg(
        F.sum("series").alias("series"),
        F.avg("history_months").alias("avg_history_months"),
        F.avg("positive_history_months").alias("avg_positive_history_months"),
        F.avg("adi").alias("avg_adi"),
        F.avg("cv_squared").alias("avg_cv_squared"),
    )
    .orderBy("cutoff", "demand_pattern")
)
show_result(score_df.orderBy("cutoff", "demand_pattern", "method"))

tsb_comparison = []
for cutoff in cutoff_dates:
    cutoff_name = cutoff.isoformat()
    intermittent_scores = {
        row["method"]: row
        for row in score_rows
        if row["cutoff"] == cutoff_name and row["demand_pattern"] == "intermittent"
    }
    if "zero" not in intermittent_scores or "tsb_a010_b010" not in intermittent_scores:
        continue
    zero_score = intermittent_scores["zero"]
    tsb_score = intermittent_scores["tsb_a010_b010"]
    tsb_comparison.append(
        {
            "cutoff": cutoff_name,
            "scope": "store_intermittent",
            "zero_wape": zero_score["wape"],
            "tsb_wape": tsb_score["wape"],
            "tsb_beats_zero_wape": (
                tsb_score["wape"] is not None
                and zero_score["wape"] is not None
                and tsb_score["wape"] < zero_score["wape"]
            ),
            "tsb_recall": tsb_score["recall"],
            "tsb_false_positive_rows": tsb_score["false_positive_rows"],
        }
    )
if tsb_comparison:
    show_result(spark.createDataFrame(tsb_comparison).orderBy("cutoff"))  # noqa: F821
else:
    print("No eligible intermittent series were found at the specified cutoffs.")

if tsb_pr_rows:
    show_result(
        spark.createDataFrame(tsb_pr_rows).orderBy("cutoff", "threshold")  # noqa: F821
    )

tsb_pr_summary = []
for cutoff in cutoff_dates:
    cutoff_name = cutoff.isoformat()
    curve = [row for row in tsb_pr_rows if row["cutoff"] == cutoff_name]
    if not curve:
        continue
    best_f1 = max(
        curve,
        key=lambda row: (
            float(row["f1"]),
            float(row["recall"]),
            float(row["precision"]),
        ),
    )
    precision_at_least_half = [row for row in curve if float(row["precision"]) >= 0.5]
    best_high_precision = (
        max(
            precision_at_least_half,
            key=lambda row: (
                float(row["recall"]),
                float(row["precision"]),
                int(row["forecast_positive_rows"]),
            ),
        )
        if precision_at_least_half
        else None
    )
    tsb_pr_summary.append(
        {
            "cutoff": cutoff_name,
            "scope": "store_intermittent",
            "roc_auc": curve[0]["roc_auc"],
            "average_precision": curve[0]["average_precision"],
            "best_f1_threshold": best_f1["threshold"],
            "best_f1_precision": best_f1["precision"],
            "best_f1_recall": best_f1["recall"],
            "best_f1_support": best_f1["forecast_positive_rows"],
            "best_recall_with_precision_at_least_50pct": (
                best_high_precision["recall"] if best_high_precision else None
            ),
            "support_at_precision_at_least_50pct": (
                best_high_precision["forecast_positive_rows"]
                if best_high_precision
                else None
            ),
        }
    )
if tsb_pr_summary:
    show_result(spark.createDataFrame(tsb_pr_summary).orderBy("cutoff"))  # noqa: F821

transition_rows = []
if len(cutoff_dates) == 2:
    first_cutoff, second_cutoff = sorted(cutoff_dates)
    first_profiles = profile_by_cutoff[first_cutoff]
    second_profiles = profile_by_cutoff[second_cutoff]
    shared_series = set(first_profiles).intersection(second_profiles)
    for first_pattern, second_pattern in sorted(
        {(first_profiles[series], second_profiles[series]) for series in shared_series}
    ):
        transition_rows.append(
            {
                "from_cutoff": first_cutoff.isoformat(),
                "to_cutoff": second_cutoff.isoformat(),
                "from_pattern": first_pattern,
                "to_pattern": second_pattern,
                "series": sum(
                    first_profiles[series] == first_pattern
                    and second_profiles[series] == second_pattern
                    for series in shared_series
                ),
            }
        )
    show_result(
        spark.createDataFrame(transition_rows).orderBy(  # noqa: F821
            "from_pattern", "to_pattern"
        )
    )

print(
    "This is research: the PR curve describes the holdout and does not calibrate a "
    "production threshold. Simulating stockout/overage costs also requires reliable "
    "historical inventory, lead times, and costs at the same time grain."
)

try:
    dbutils.notebook.exit(  # noqa: F821
        json.dumps(
            {
                "status": "research_only_no_forecast_published",
                "profile_summary": profile_summary_rows,
                "scores": score_rows,
                "tsb_precision_recall_summary": tsb_pr_summary,
                "profile_transitions": transition_rows,
                "tsb_intermittent_comparison": tsb_comparison,
            }
        )
    )
except NameError:
    pass
