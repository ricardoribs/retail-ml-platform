"""Mixture-model backtest without Spark, I/O, or holdout tuning."""

from __future__ import annotations

from datetime import date

import pytest

from ml.demand_forecasting.src.intermittent_bayes import (
    BASELINE_METHODS,
    HURDLE_METHODS,
    build_hurdle_forecasts,
)
from ml.demand_forecasting.src.modeling import shift_months


def _months(origin: date, quantity: float = 0.0) -> dict[date, float]:
    return {shift_months(origin, -i): quantity for i in range(1, 25)}


def test_frequent_series_has_positive_median_and_conserves_keys():
    origin = date(2026, 9, 1)
    history = {(1, 10): _months(origin, 3.0)}
    groups = {(1, 10): (10, "motor")}
    targets = (origin, shift_months(origin, 1), shift_months(origin, 2))
    result = build_hurdle_forecasts(
        series_groups=groups, series_history=history, target_months=targets
    )
    assert set(result) == {*BASELINE_METHODS, *HURDLE_METHODS}
    assert all(len(predictions) == 3 for predictions in result.values())
    assert result["hurdle_median"][(origin, 1, 10)] == 3
    assert result["hurdle_calendar_median"][(origin, 1, 10)] == 3
    assert result["hurdle_calendar_mean"][(origin, 1, 10)] == pytest.approx(3)


def test_calendar_changes_probability_without_using_target_sales():
    origin = date(2026, 9, 1)
    months = _months(origin)
    for lag in (1, 2, 3, 4, 12, 24):
        months[shift_months(origin, -lag)] = 3.0
    result = build_hurdle_forecasts(
        series_groups={(1, 10): (10, "motor")},
        series_history={(1, 10): months},
        target_months=(origin,),
    )
    key = (origin, 1, 10)
    assert result["hurdle_median"][key] == 0.0
    assert result["hurdle_calendar_median"][key] == 3.0
    assert result["hurdle_calendar_mean"][key] == pytest.approx(2.5)
    assert result["seasonal_series"][key] == 3.0


def test_no_sale_history_does_not_invent_positive_quantity():
    origin = date(2026, 9, 1)
    result = build_hurdle_forecasts(
        series_groups={(1, 10): (10, "motor")},
        series_history={(1, 10): _months(origin)},
        target_months=(origin,),
    )
    assert all(predictions[(origin, 1, 10)] == 0 for predictions in result.values())


def test_pooling_can_supply_magnitude_but_not_force_positive_median():
    origin = date(2026, 9, 1)
    groups = {(1, 10): (10, "motor"), (1, 20): (20, "motor")}
    histories = {(1, 10): _months(origin, 4.0), (1, 20): _months(origin)}
    result = build_hurdle_forecasts(
        series_groups=groups, series_history=histories, target_months=(origin,)
    )
    silent = (origin, 1, 20)
    assert result["hurdle_median"][silent] == 0
    assert result["hurdle_calendar_mean"][silent] > 0


def test_future_or_invalid_history_and_nonconsecutive_targets_are_rejected():
    origin = date(2026, 9, 1)
    groups = {(1, 10): (10, "motor")}
    history = {(1, 10): _months(origin)}
    history[(1, 10)][origin] = 1000.0
    with pytest.raises(ValueError, match="futuro"):
        build_hurdle_forecasts(
            series_groups=groups, series_history=history, target_months=(origin,)
        )
    del history[(1, 10)][origin]
    history[(1, 10)][shift_months(origin, -1)] = -1.0
    with pytest.raises(ValueError, match="invalid quantity"):
        build_hurdle_forecasts(
            series_groups=groups, series_history=history, target_months=(origin,)
        )
    history[(1, 10)][shift_months(origin, -1)] = 0.0
    with pytest.raises(ValueError, match="consecutivos"):
        build_hurdle_forecasts(
            series_groups=groups,
            series_history=history,
            target_months=(origin, shift_months(origin, 2)),
        )
