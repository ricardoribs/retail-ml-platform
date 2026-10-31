"""Selective forecast for intermittent demand with temporal calibration.

This candidate starts from the existing hurdle expectation, but publishes it
only where the estimated sale probability exceeds a threshold. The caller
chooses the threshold using historical windows before the test; this module
receives no future actuals and performs no I/O."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

from ml.demand_forecasting.src.hierarchical import ForecastKey, GroupKey, SeriesKey
from ml.demand_forecasting.src.intermittent_bayes import build_hurdle_forecasts
from ml.demand_forecasting.src.modeling import shift_months

SELECTIVE_MODEL_NAME = "selective_hurdle_nested"
PROBABILITY_THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
OCCURRENCE_F_BETA = 2.0
PRIOR_MONTHS = 12


@dataclass(frozen=True)
class ThresholdSelection:
    """Operating point selected only from inner windows."""

    threshold: float
    precision: float
    recall: float
    fbeta: float
    predicted_positive_rows: int
    true_positive_rows: int
    false_positive_rows: int
    false_negative_rows: int

    @property
    def has_detected_sales(self) -> bool:
        """Indicate whether the selected operating point detected any sales in calibration."""
        return self.true_positive_rows > 0


def _validate_targets(target_months: Sequence[date]) -> tuple[date, ...]:
    targets = tuple(target_months)
    if not targets or any(month.day != 1 for month in targets):
        raise ValueError("Target months must be the first day of each month")
    expected = tuple(shift_months(targets[0], offset) for offset in range(len(targets)))
    if targets != expected:
        raise ValueError("Target months must be consecutive and ordered")
    return targets


def _validate_history(
    series_groups: Mapping[SeriesKey, GroupKey],
    series_history: Mapping[SeriesKey, Mapping[date, float]],
    origin: date,
) -> None:
    if not series_groups or set(series_groups) != set(series_history):
        raise ValueError("History must exactly cover the cohort series")
    for series, group in series_groups.items():
        if series[1] != group[0]:
            raise ValueError("The series location does not match the group")
        for month, quantity in series_history[series].items():
            if month.day != 1 or month >= origin:
                raise ValueError("History contains a future or invalid month")
            if not math.isfinite(float(quantity)) or quantity < 0:
            raise ValueError("History contains an invalid quantity")


def build_occurrence_probabilities(
    *,
    series_groups: Mapping[SeriesKey, GroupKey],
    series_history: Mapping[SeriesKey, Mapping[date, float]],
    target_months: Sequence[date],
) -> dict[ForecastKey, float]:
    """Estimate prior sales probability by SKU-location and month.

    The local rate over the latest twelve months is smoothed by product and
    category-location. The seasonal adjustment uses only historical rates for
    the calendar month and is bounded to avoid extrapolation.
    """
    targets = _validate_targets(target_months)
    origin = targets[0]
    _validate_history(series_groups, series_history, origin)

    last12 = tuple(shift_months(origin, -offset) for offset in range(1, 13))
    last24 = tuple(shift_months(origin, -offset) for offset in range(1, 25))
    product_recent: dict[int, list[float]] = defaultdict(list)
    group_recent: dict[GroupKey, list[float]] = defaultdict(list)
    group_calendar: dict[tuple[GroupKey, int], list[float]] = defaultdict(list)
    group_all: dict[GroupKey, list[float]] = defaultdict(list)

    for series, history in series_history.items():
        product_id = series[0]
        group = series_groups[series]
        for month in last12:
            if month in history:
                quantity = float(history[month])
                product_recent[product_id].append(quantity)
                group_recent[group].append(quantity)
        for month in last24:
            if month in history:
                quantity = float(history[month])
                group_calendar[(group, month.month)].append(quantity)
                group_all[group].append(quantity)

    probabilities: dict[ForecastKey, float] = {}
    for series, history in series_history.items():
        product_id, location_id = series
        group = series_groups[series]
        recent = [float(history[month]) for month in last12 if month in history]
        product_rate = (
            sum(value > 0 for value in product_recent[product_id])
            / len(product_recent[product_id])
            if product_recent[product_id]
            else 0.0
        )
        group_rate = (
            sum(value > 0 for value in group_recent[group]) / len(group_recent[group])
            if group_recent[group]
            else 0.0
        )
        prior_rate = (product_rate + group_rate) / 2
        local_probability = (
            sum(value > 0 for value in recent) + PRIOR_MONTHS * prior_rate
        ) / (len(recent) + PRIOR_MONTHS)
        overall_group_rate = (
            sum(value > 0 for value in group_all[group]) / len(group_all[group])
            if group_all[group]
            else 0.0
        )
        for month in targets:
            calendar_values = group_calendar[(group, month.month)]
            calendar_rate = (
                sum(value > 0 for value in calendar_values) / len(calendar_values)
                if calendar_values
                else 0.0
            )
            season_factor = (
                min(2.0, max(0.5, calendar_rate / overall_group_rate))
                if overall_group_rate > 0 and calendar_values
                else 1.0
            )
            probabilities[(month, product_id, location_id)] = min(
                1.0, local_probability * season_factor
            )
    return probabilities


def threshold_predictions(
    *,
    expected_demand: Mapping[ForecastKey, float],
    probabilities: Mapping[ForecastKey, float],
    threshold: float,
) -> dict[ForecastKey, float]:
    """Apply a threshold without changing expected values or probabilities."""
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("threshold must be in the interval [0, 1]")
    if set(expected_demand) != set(probabilities):
        raise ValueError("Expected demand and probability must cover the same keys")
    predicted: dict[ForecastKey, float] = {}
    for key, expected in expected_demand.items():
        numeric_expected = float(expected)
        probability = float(probabilities[key])
        if (
            not math.isfinite(numeric_expected)
            or numeric_expected < 0
            or not math.isfinite(probability)
            or not 0 <= probability <= 1
        ):
            raise ValueError("Invalid expectation or probability")
        predicted[key] = numeric_expected if probability >= threshold else 0.0
    return predicted


def build_selective_candidates(
    *,
    series_groups: Mapping[SeriesKey, GroupKey],
    series_history: Mapping[SeriesKey, Mapping[date, float]],
    target_months: Sequence[date],
    thresholds: Sequence[float] = PROBABILITY_THRESHOLDS,
) -> dict[float, dict[ForecastKey, float]]:
    """Build forecasts for a predefined set of thresholds."""
    if not thresholds:
        raise ValueError("Provide at least one threshold")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError("Thresholds must be distinct")
    probabilities = build_occurrence_probabilities(
        series_groups=series_groups,
        series_history=series_history,
        target_months=target_months,
    )
    conditional_quantity = build_conditional_quantity_forecast(
        series_groups=series_groups,
        series_history=series_history,
        target_months=target_months,
        probabilities=probabilities,
    )
    return {
        float(threshold): threshold_predictions(
            expected_demand=conditional_quantity,
            probabilities=probabilities,
            threshold=float(threshold),
        )
        for threshold in thresholds
    }


def build_conditional_quantity_forecast(
    *,
    series_groups: Mapping[SeriesKey, GroupKey],
    series_history: Mapping[SeriesKey, Mapping[date, float]],
    target_months: Sequence[date],
    probabilities: Mapping[ForecastKey, float] | None = None,
) -> dict[ForecastKey, float]:
    """Recover E[quantity | sale] from the hurdle expectation.

    The existing hurdle candidate produces ``P(sale) × E[quantity | sale]``.
    Division is safe because expectation and probability are both zero when no
    positive evidence is available.
    """
    occurrence_probability = (
        dict(probabilities)
        if probabilities is not None
        else build_occurrence_probabilities(
            series_groups=series_groups,
            series_history=series_history,
            target_months=target_months,
        )
    )
    expected_demand = build_hurdle_forecasts(
        series_groups=series_groups,
        series_history=series_history,
        target_months=target_months,
    )["hurdle_calendar_mean"]
    if set(expected_demand) != set(occurrence_probability):
        raise ValueError("Expected values and probabilities must cover the same keys")
    conditional_quantity: dict[ForecastKey, float] = {}
    for key, probability in occurrence_probability.items():
        numeric_probability = float(probability)
        expected = float(expected_demand[key])
        if (
            not math.isfinite(numeric_probability)
            or not 0 <= numeric_probability <= 1
            or not math.isfinite(expected)
            or expected < 0
        ):
            raise ValueError("Invalid expectation or probability")
        if numeric_probability == 0 and expected > 0:
            raise ValueError("Positive expectation with zero sales probability")
        conditional_quantity[key] = (
            expected / numeric_probability if numeric_probability > 0 else 0.0
        )
    return conditional_quantity


def select_threshold_from_history(
    *,
    actual: Mapping[ForecastKey, float],
    probabilities: Mapping[ForecastKey, float],
    thresholds: Sequence[float] = PROBABILITY_THRESHOLDS,
    beta: float = OCCURRENCE_F_BETA,
) -> ThresholdSelection:
    """Choose a threshold by F-beta without using magnitude or WAPE.

    ``beta=2`` gives more weight to recall, an explicit assumption about the cost
    of stockouts. WAPE remains reserved for final evaluation of the complete forecast.
    """
    if not actual or not probabilities or not thresholds:
        raise ValueError("Actuals, probabilities, and thresholds cannot be empty")
    if set(actual) != set(probabilities):
        raise ValueError("Actuals and probabilities must cover the same keys")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    normalized_thresholds = tuple(float(threshold) for threshold in thresholds)
    if len(set(normalized_thresholds)) != len(normalized_thresholds):
        raise ValueError("Thresholds must be distinct")
    if any(
        not math.isfinite(threshold) or not 0 <= threshold <= 1
        for threshold in normalized_thresholds
    ):
        raise ValueError("Thresholds must be in the interval [0, 1]")
    ordered_keys = tuple(sorted(actual))
    if any(not math.isfinite(float(value)) or value < 0 for value in actual.values()):
        raise ValueError("Actual values must be finite and non-negative")
    if any(
        not math.isfinite(float(value)) or not 0 <= float(value) <= 1
        for value in probabilities.values()
    ):
        raise ValueError("Probabilities must be finite and in [0, 1]")

    scores: list[ThresholdSelection] = []
    beta_squared = beta**2
    for threshold in normalized_thresholds:
        actual_events = [float(actual[key]) > 0 for key in ordered_keys]
        predicted_events = [
            float(probabilities[key]) >= threshold for key in ordered_keys
        ]
        true_positive_rows = sum(
            observed and predicted
            for observed, predicted in zip(actual_events, predicted_events, strict=True)
        )
        false_positive_rows = sum(
            not observed and predicted
            for observed, predicted in zip(actual_events, predicted_events, strict=True)
        )
        false_negative_rows = sum(
            observed and not predicted
            for observed, predicted in zip(actual_events, predicted_events, strict=True)
        )
        predicted_positive_rows = true_positive_rows + false_positive_rows
        actual_positive_rows = true_positive_rows + false_negative_rows
        precision = (
            true_positive_rows / predicted_positive_rows
            if predicted_positive_rows
            else 0.0
        )
        recall = (
            true_positive_rows / actual_positive_rows if actual_positive_rows else 0.0
        )
        denominator = beta_squared * precision + recall
        fbeta = (
            (1 + beta_squared) * precision * recall / denominator
            if denominator
            else 0.0
        )
        scores.append(
            ThresholdSelection(
                threshold=threshold,
                precision=precision,
                recall=recall,
                fbeta=fbeta,
                predicted_positive_rows=predicted_positive_rows,
                true_positive_rows=true_positive_rows,
                false_positive_rows=false_positive_rows,
                false_negative_rows=false_negative_rows,
            )
        )
    return max(
        scores,
        key=lambda score: (
            score.fbeta,
            score.recall,
            score.precision,
            -score.predicted_positive_rows,
            -score.threshold,
        ),
    )
