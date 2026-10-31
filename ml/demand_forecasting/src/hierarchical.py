"""Transparent hierarchical candidates for backtesting, with no external state.

All forecast inputs must be known before the first target month. Realized
holdout values may only be passed to `evaluate_forecasts`."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date

from ml.demand_forecasting.src.metrics import calculate_metrics
from ml.demand_forecasting.src.modeling import shift_months

SeriesKey = tuple[int, int]  # product, location
GroupKey = tuple[int, str]  # local, categoria
ForecastKey = tuple[date, int, int]  # month, product, location

FORECAST_METHODS = (
    "seasonal_uniform",
    "trend_uniform",
    "trend_local",
    "trend_pooled",
    "ma3_uniform",
)


def _nonnegative_finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _validate_targets(target_months: Sequence[date]) -> tuple[date, ...]:
    targets = tuple(target_months)
    if not targets or any(month.day != 1 for month in targets):
        raise ValueError("target_months must contain first days of months")
    if targets != tuple(shift_months(targets[0], i) for i in range(len(targets))):
        raise ValueError("target_months must be consecutive and ordered")
    return targets


def build_hierarchical_forecasts(
    series_groups: Mapping[SeriesKey, GroupKey],
    group_history: Mapping[GroupKey, Mapping[date, float]],
    local_last12: Mapping[SeriesKey, float],
    pooled_last12: Mapping[int, float],
    target_months: Sequence[date],
) -> dict[str, dict[ForecastKey, float]]:
    """Forecast groups from seasonal/MA3 history and allocate by historical weights.

    Annual trend uses the latest 12 months divided by the prior 12, clipped to
    [0.5, 2]. If fewer than 24 months are available or the denominator is not
    positive, use 1. A missing or zero seasonal month falls back to MA3. Weights
    receive fixed +1 smoothing. No parameters are tuned on the holdouts.
    """
    targets = _validate_targets(target_months)
    origin = targets[0]
    if not series_groups:
        raise ValueError("No series to forecast")

    members: dict[GroupKey, list[SeriesKey]] = defaultdict(list)
    for series, group in series_groups.items():
        if series[1] != group[0]:
            raise ValueError("Series location differs from the group location")
        members[group].append(series)

    if set(group_history) != set(members):
        raise ValueError("Group history does not match the series")
    for group, history in group_history.items():
        for month, quantity in history.items():
            if month.day != 1 or month >= origin:
                raise ValueError(f"Future history or invalid month in group {group}")
            _nonnegative_finite(quantity, "historical demand")
    for quantity in local_last12.values():
        _nonnegative_finite(quantity, "local weight")
    for quantity in pooled_last12.values():
        _nonnegative_finite(quantity, "global weight")

    recent12 = tuple(shift_months(origin, -i) for i in range(1, 13))
    previous12 = tuple(shift_months(origin, -i) for i in range(13, 25))
    forecasts: dict[str, dict[ForecastKey, float]] = {
        method: {} for method in FORECAST_METHODS
    }

    for group, group_members in members.items():
        history = group_history[group]
        recent3 = tuple(shift_months(origin, -i) for i in range(1, 4))
        ma3 = sum(history.get(month, 0.0) for month in recent3) / 3.0
        if all(month in history for month in (*recent12, *previous12)):
            denominator = sum(history[month] for month in previous12)
            ratio = (
                sum(history[month] for month in recent12) / denominator
                if denominator > 0
                else 1.0
            )
            ratio = min(2.0, max(0.5, ratio))
        else:
            ratio = 1.0

        weight_options = {
            "uniform": {series: 1.0 for series in group_members},
            "local": {
                series: 1.0 + local_last12.get(series, 0.0) for series in group_members
            },
            "pooled": {
                series: 1.0 + pooled_last12.get(series[0], 0.0)
                for series in group_members
            },
        }
        weight_totals = {
            name: sum(weights.values()) for name, weights in weight_options.items()
        }

        for month in targets:
            last_year = history.get(shift_months(month, -12), 0.0)
            seasonal = last_year if last_year > 0 else ma3
            group_predictions = {
                "seasonal_uniform": (seasonal, "uniform"),
                "trend_uniform": (seasonal * ratio, "uniform"),
                "trend_local": (seasonal * ratio, "local"),
                "trend_pooled": (seasonal * ratio, "pooled"),
                "ma3_uniform": (ma3, "uniform"),
            }
            for method, (group_qty, weighting) in group_predictions.items():
                for product_id, location_id in group_members:
                    series = (product_id, location_id)
                    forecasts[method][(month, product_id, location_id)] = (
                        group_qty
                        * weight_options[weighting][series]
                        / weight_totals[weighting]
                    )
    return forecasts


def evaluate_forecasts(
    actual: Mapping[ForecastKey, float],
    predicted: Mapping[ForecastKey, float],
    series_groups: Mapping[SeriesKey, GroupKey],
) -> dict[str, dict[str, float | int]]:
    """Measure the same rows at the required and aggregate grains."""
    if not actual or set(actual) != set(predicted):
        raise ValueError("Actuals and forecasts must have the same non-empty keys")
    if {key[1:] for key in actual} != set(series_groups):
        raise ValueError("Series keys do not match the holdout cohort")

    actual_groups: dict[str, dict[tuple[object, ...], float]] = {
        name: defaultdict(float)
        for name in ("sku_location", "category_location", "location", "total_month")
    }
    predicted_groups: dict[str, dict[tuple[object, ...], float]] = {
        name: defaultdict(float) for name in actual_groups
    }
    for key in actual:
        month, product_id, location_id = key
        group = series_groups[(product_id, location_id)]
        buckets = {
            "sku_location": key,
            "category_location": (month, *group),
            "location": (month, location_id),
            "total_month": (month,),
        }
        actual_qty = _nonnegative_finite(actual[key], "realizado")
        predicted_qty = _nonnegative_finite(predicted[key], "forecast")
        for name, bucket in buckets.items():
            actual_groups[name][bucket] += actual_qty
            predicted_groups[name][bucket] += predicted_qty

    return {
        name: calculate_metrics(
            [values[key] for key in sorted(values)],
            [predicted_groups[name][key] for key in sorted(values)],
        )
        for name, values in actual_groups.items()
    }
