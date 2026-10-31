"""Two-stage forecast candidate: location, then product-location.

This candidate aims to avoid an independent series-level decision zeroing
most of the portfolio when aggregate location demand remains seasonal. It
first estimates each store/DC total, then allocates that total across products
at the same location using shares known before the target month.

This module is deliberately separate from the operational job. Promotion
requires a recursive backtest to show improvement at the required
product-location grain and in planning totals."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date

from ml.demand_forecasting.src.hierarchical import ForecastKey, SeriesKey
from ml.demand_forecasting.src.modeling import shift_months

LOCATION_ALLOCATION_METHODS = (
    "location_seasonal_share",
    "location_ma3_share",
)


def _validate_target_months(target_months: Sequence[date]) -> tuple[date, ...]:
    targets = tuple(target_months)
    if not targets or any(month.day != 1 for month in targets):
        raise ValueError("target_months must contain first days of months")
    expected_targets = tuple(
        shift_months(targets[0], offset) for offset in range(len(targets))
    )
    if targets != expected_targets:
        raise ValueError("target_months must be consecutive and ordered")
    return targets


def _validate_history(
    series_history: Mapping[SeriesKey, Mapping[date, float]], origin: date
) -> None:
    if not series_history:
        raise ValueError("No series were provided")
    for series, history in series_history.items():
        if not history:
            raise ValueError(f"History is empty for series {series}")
        for month, quantity in history.items():
            if month.day != 1 or month >= origin:
                raise ValueError(f"Future history or invalid month in series {series}")
            numeric_quantity = float(quantity)
            if not math.isfinite(numeric_quantity) or numeric_quantity < 0:
                raise ValueError(f"Invalid historical quantity in series {series}")


def _allocation_shares(
    *,
    members: Sequence[SeriesKey],
    series_history: Mapping[SeriesKey, Mapping[date, float]],
    origin: date,
    target_month: date,
) -> dict[SeriesKey, float]:
    """Combine the recent share with the share from the same month last year.

    The 50/50 rule is fixed before the holdouts. If one basis is unavailable, use
    the other; if neither has sales, allocate equally.
    """
    recent_months = tuple(shift_months(origin, -offset) for offset in range(1, 13))
    prior_year = shift_months(target_month, -12)
    recent_totals = {
        series: sum(series_history[series].get(month, 0.0) for month in recent_months)
        for series in members
    }
    prior_totals = {
        series: series_history[series].get(prior_year, 0.0) for series in members
    }
    recent_total = sum(recent_totals.values())
    prior_total = sum(prior_totals.values())

    shares: dict[SeriesKey, float] = {}
    for series in members:
        recent_share = (
            recent_totals[series] / recent_total if recent_total > 0 else None
        )
        prior_share = prior_totals[series] / prior_total if prior_total > 0 else None
        if recent_share is not None and prior_share is not None:
            shares[series] = (recent_share + prior_share) / 2.0
        elif recent_share is not None:
            shares[series] = recent_share
        elif prior_share is not None:
            shares[series] = prior_share
        else:
            shares[series] = 1.0 / len(members)
    return shares


def build_location_allocation_forecasts(
    *,
    series_history: Mapping[SeriesKey, Mapping[date, float]],
    target_months: Sequence[date],
) -> dict[str, dict[ForecastKey, float]]:
    """Forecast the store/DC total and allocate it to each product-location pair.

    ``location_seasonal_share`` uses demand from the same location and month in
    the previous year, with recursive MA3 only when the seasonal month is missing.
    ``location_ma3_share`` is the non-seasonal benchmark. Both preserve the exact
    forecast total for each location and month.
    """
    targets = _validate_target_months(target_months)
    origin = targets[0]
    _validate_history(series_history, origin)

    members_by_location: dict[int, list[SeriesKey]] = defaultdict(list)
    location_history: dict[int, dict[date, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for series, history in series_history.items():
        _product_id, location_id = series
        members_by_location[location_id].append(series)
        for month, quantity in history.items():
            location_history[location_id][month] += float(quantity)

    forecasts: dict[str, dict[ForecastKey, float]] = {
        method: {} for method in LOCATION_ALLOCATION_METHODS
    }
    recursive_totals = {
        method: {
            location_id: dict(history)
            for location_id, history in location_history.items()
        }
        for method in LOCATION_ALLOCATION_METHODS
    }

    for target_month in targets:
        prior_year = shift_months(target_month, -12)
        for location_id, members in members_by_location.items():
            shares = _allocation_shares(
                members=members,
                series_history=series_history,
                origin=origin,
                target_month=target_month,
            )
            for method in LOCATION_ALLOCATION_METHODS:
                history = recursive_totals[method][location_id]
                recent3 = (
                    sum(
                        history.get(shift_months(target_month, -offset), 0.0)
                        for offset in range(1, 4)
                    )
                    / 3.0
                )
                seasonal_total = history.get(prior_year)
                predicted_total = (
                    seasonal_total
                    if method == "location_seasonal_share"
                    and seasonal_total is not None
                    else recent3
                )
                if not math.isfinite(predicted_total) or predicted_total < 0:
                    raise ValueError("Invalid location forecast total")
                history[target_month] = predicted_total
                for product_id, member_location_id in members:
                    forecast_key = (target_month, product_id, member_location_id)
                    forecasts[method][forecast_key] = (
                        predicted_total * shares[(product_id, member_location_id)]
                    )
    return forecasts
