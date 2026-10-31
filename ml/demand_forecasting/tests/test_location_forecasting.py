"""Contratos puros do forecast sazonal agregado por localidade."""

from datetime import date

import pytest

from ml.demand_forecasting.src.location_forecasting import (
    LOCATION_SEASONAL_MODEL_NAME,
    LOCATION_SEASONAL_MODEL_VERSION,
    build_location_seasonal_forecasts,
)
from ml.demand_forecasting.src.modeling import shift_months


def test_location_seasonal_forecast_uses_same_month_last_year_and_fallback_ma3():
    origin = date(2026, 9, 1)
    history = {
        10: {shift_months(origin, -offset): float(offset) for offset in range(1, 11)}
    }
    history[10][date(2025, 9, 1)] = 40.0

    forecasts = build_location_seasonal_forecasts(
        location_history=history,
        target_months=(origin, shift_months(origin, 1)),
    )

    assert LOCATION_SEASONAL_MODEL_NAME == "location_seasonal_naive"
    assert LOCATION_SEASONAL_MODEL_VERSION == "location_seasonal_naive_v1"
    assert forecasts[(origin, 10)] == pytest.approx(40.0)
    assert forecasts[(shift_months(origin, 1), 10)] == pytest.approx(43.0 / 3.0)


def test_location_seasonal_forecast_is_recursive_and_rejects_invalid_history():
    origin = date(2026, 9, 1)
    history = {10: {shift_months(origin, -offset): 3.0 for offset in range(1, 4)}}
    forecasts = build_location_seasonal_forecasts(
        location_history=history,
        target_months=(origin, shift_months(origin, 1), shift_months(origin, 2)),
    )
    assert forecasts[(origin, 10)] == pytest.approx(3.0)
    assert forecasts[(shift_months(origin, 1), 10)] == pytest.approx(3.0)
    assert forecasts[(shift_months(origin, 2), 10)] == pytest.approx(3.0)

    history[10][origin] = 1.0
    with pytest.raises(ValueError, match="futuro"):
        build_location_seasonal_forecasts(
            location_history=history,
            target_months=(origin,),
        )
    with pytest.raises(ValueError, match="consecutivo"):
        build_location_seasonal_forecasts(
            location_history={10: {shift_months(origin, -1): 1.0}},
            target_months=(origin, date(2026, 11, 1)),
        )
