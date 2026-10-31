"""Empirical candidate for intermittent monthly sales, for backtesting only.

All data used for a forecast precede the first target month. A mixture with
point mass at zero produces a median suited to absolute error/WAPE; the
alternative conditional mean helps measure the tradeoff in aggregate volume.
This module does not approve either candidate for publication."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date

from ml.demand_forecasting.src.hierarchical import ForecastKey, GroupKey, SeriesKey
from ml.demand_forecasting.src.modeling import shift_months

HURDLE_METHODS = (
    "hurdle_median",
    "hurdle_calendar_median",
    "hurdle_calendar_mean",
)
BASELINE_METHODS = ("zero", "ma3_frozen", "seasonal_series")
PRIOR_MONTHS = 12
LOCAL_MAGNITUDE_PRIOR = 6


def mixture_median(probability: float, positive_values: Sequence[float]) -> float:
    """Median of a distribution with mass 1-p at zero and a positive sample."""
    if probability <= 0.5 or not positive_values:
        return 0.0
    quantile = 1.0 - 0.5 / probability
    sorted_values = sorted(positive_values)
    index = max(0, math.ceil(quantile * len(sorted_values)) - 1)
    return sorted_values[index]


def _sale_rate(values: Sequence[float]) -> float:
    return sum(value > 0 for value in values) / len(values) if values else 0.0


def _validate_history(
    series_groups: Mapping[SeriesKey, GroupKey],
    series_history: Mapping[SeriesKey, Mapping[date, float]],
    origin: date,
) -> None:
    if not series_groups or set(series_groups) != set(series_history):
        raise ValueError("History must exactly cover the cohort series")
    for series, group in series_groups.items():
        if series[1] != group[0]:
            raise ValueError("Series location differs from the group location")
        for month, quantity in series_history[series].items():
            if month.day != 1 or month >= origin:
                raise ValueError("History contains a future or invalid month")
            if not math.isfinite(float(quantity)) or quantity < 0:
                raise ValueError("History contains an invalid quantity")


def build_hurdle_forecasts(
    *,
    series_groups: Mapping[SeriesKey, GroupKey],
    series_history: Mapping[SeriesKey, Mapping[date, float]],
    target_months: Sequence[date],
) -> dict[str, dict[ForecastKey, float]]:
    """Forecast observed sales without selecting thresholds from holdouts.

    The local sales probability is smoothed using 12 months of product-level
    evidence (all stores) and category-by-store evidence. Seasonality multiplies
    this probability by the ratio of the calendar-month rate to the overall
    category-by-store rate, clipped to [0.5, 2]. Positive quantity uses the
    product distribution, or category-by-store distribution when examples are
    insufficient; the mean receives 6 prior examples as regularization. All
    values are fixed before the test.
    """
    targets = tuple(target_months)
    if not targets or any(month.day != 1 for month in targets):
        raise ValueError("Target months must be first days of months")
    if targets != tuple(shift_months(targets[0], i) for i in range(len(targets))):
        raise ValueError("Target months must be consecutive and ordered")
    origin = targets[0]
    _validate_history(series_groups, series_history, origin)

    last12 = tuple(shift_months(origin, -i) for i in range(1, 13))
    last24 = tuple(shift_months(origin, -i) for i in range(1, 25))
    last3 = last12[:3]
    product_recent: dict[int, list[float]] = defaultdict(list)
    group_recent: dict[GroupKey, list[float]] = defaultdict(list)
    product_positive: dict[int, list[float]] = defaultdict(list)
    group_positive: dict[GroupKey, list[float]] = defaultdict(list)
    group_calendar: dict[tuple[GroupKey, int], list[float]] = defaultdict(list)
    group_all: dict[GroupKey, list[float]] = defaultdict(list)

    for series, history in series_history.items():
        group = series_groups[series]
        product_id = series[0]
        for month in last12:
            if month in history:
                quantity = history[month]
                product_recent[product_id].append(quantity)
                group_recent[group].append(quantity)
        for month in last24:
            if month in history:
                quantity = history[month]
                group_calendar[(group, month.month)].append(quantity)
                group_all[group].append(quantity)
                if quantity > 0:
                    product_positive[product_id].append(quantity)
                    group_positive[group].append(quantity)

    forecasts: dict[str, dict[ForecastKey, float]] = {
        name: {} for name in (*BASELINE_METHODS, *HURDLE_METHODS)
    }
    for series, history in series_history.items():
        group = series_groups[series]
        product_id, location_id = series
        recent = [history[month] for month in last12 if month in history]
        product_rate = _sale_rate(product_recent[product_id])
        group_rate = _sale_rate(group_recent[group])
        prior_rate = (product_rate + group_rate) / 2
        local_probability = (
            sum(value > 0 for value in recent) + PRIOR_MONTHS * prior_rate
        ) / (len(recent) + PRIOR_MONTHS)

        local_positive = [
            history[month]
            for month in last24
            if month in history and history[month] > 0
        ]
        pooled_positive = (
            product_positive[product_id]
            if product_positive[product_id]
            else group_positive[group]
        )
        reference_mean = (
            sum(pooled_positive) / len(pooled_positive) if pooled_positive else 0.0
        )
        positive_mean = (
            sum(local_positive) + LOCAL_MAGNITUDE_PRIOR * reference_mean
        ) / (len(local_positive) + LOCAL_MAGNITUDE_PRIOR)
        positive_distribution = (
            local_positive
            if len(local_positive) >= LOCAL_MAGNITUDE_PRIOR
            else pooled_positive
        )
        ma3 = sum(history.get(month, 0.0) for month in last3) / 3
        overall_group_rate = _sale_rate(group_all[group])

        for month in targets:
            key = (month, product_id, location_id)
            forecasts["zero"][key] = 0.0
            forecasts["ma3_frozen"][key] = ma3
            prior_year = history.get(shift_months(month, -12))
            forecasts["seasonal_series"][key] = (
                prior_year if prior_year is not None else ma3
            )

            calendar_rate = _sale_rate(group_calendar[(group, month.month)])
            season_factor = (
                min(2.0, max(0.5, calendar_rate / overall_group_rate))
                if overall_group_rate > 0 and group_calendar[(group, month.month)]
                else 1.0
            )
            calendar_probability = min(1.0, local_probability * season_factor)
            forecasts["hurdle_median"][key] = mixture_median(
                local_probability, positive_distribution
            )
            forecasts["hurdle_calendar_median"][key] = mixture_median(
                calendar_probability, positive_distribution
            )
            forecasts["hurdle_calendar_mean"][key] = (
                calendar_probability * positive_mean
            )
    return forecasts
