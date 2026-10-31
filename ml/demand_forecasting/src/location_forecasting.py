"""Seasonal forecast of aggregate demand by store/DC.

This model has a different purpose from the product-location forecast: it
predicts each location's planning volume. It uses the same month from the
previous year, with a recursive 3-month moving average when no seasonal
observation is available. The rule passed three backtests before entering
the operational workflow."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date

from ml.demand_forecasting.src.modeling import shift_months

LOCATION_SEASONAL_MODEL_NAME = "location_seasonal_naive"
LOCATION_SEASONAL_MODEL_VERSION = "location_seasonal_naive_v1"


def _validate_target_months(target_months: Sequence[date]) -> tuple[date, ...]:
    targets = tuple(target_months)
    if not targets or any(month.day != 1 for month in targets):
        raise ValueError("target_months must contain first days of months")
    expected = tuple(shift_months(targets[0], offset) for offset in range(len(targets)))
    if targets != expected:
        raise ValueError("target_months must be consecutive and ordered")
    return targets


def build_location_seasonal_forecasts(
    *,
    location_history: Mapping[int, Mapping[date, float]],
    target_months: Sequence[date],
) -> dict[tuple[date, int], float]:
    """Generate recursive forecasts by location without using future data.

    The return key is ``(forecast_month, location_id)``. Each monthly forecast
    is added to its own history before the next horizon so that the MA3 fallback
    does not read future actuals.
    """
    targets = _validate_target_months(target_months)
    origin = targets[0]
    if not location_history:
        raise ValueError("No locations were provided")

    recursive_history: dict[int, dict[date, float]] = {}
    for location_id, history in location_history.items():
        if not history:
            raise ValueError(f"History is empty for location {location_id}")
        recursive_history[location_id] = {}
        for month, quantity in history.items():
            if month.day != 1 or month >= origin:
                raise ValueError(
                    f"Future history or invalid month for location {location_id}"
                )
            numeric_quantity = float(quantity)
            if not math.isfinite(numeric_quantity) or numeric_quantity < 0:
                raise ValueError(
                    f"Invalid historical quantity for location {location_id}"
                )
            recursive_history[location_id][month] = numeric_quantity

    forecasts: dict[tuple[date, int], float] = {}
    for target_month in targets:
        previous_year_month = shift_months(target_month, -12)
        for location_id, history in recursive_history.items():
            seasonal_value = history.get(previous_year_month)
            if seasonal_value is None:
                seasonal_value = (
                    sum(
                        history.get(shift_months(target_month, -offset), 0.0)
                        for offset in range(1, 4)
                    )
                    / 3.0
                )
            forecast = max(float(seasonal_value), 0.0)
            history[target_month] = forecast
            forecasts[(target_month, location_id)] = forecast
    return forecasts
