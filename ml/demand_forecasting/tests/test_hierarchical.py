"""Spark-free contracts for the exploratory hierarchical backtest."""

from __future__ import annotations

from datetime import date

import pytest

from ml.demand_forecasting.src.hierarchical import (
    FORECAST_METHODS,
    build_hierarchical_forecasts,
    evaluate_forecasts,
)
from ml.demand_forecasting.src.location_allocation import (
    LOCATION_ALLOCATION_METHODS,
    build_location_allocation_forecasts,
)
from ml.demand_forecasting.src.modeling import shift_months


def _case():
    first_target = date(2026, 9, 1)
    group = (10, "motores")
    series_groups = {(1, 10): group, (2, 10): group}
    history = {
        shift_months(first_target, -i): (12.0 if i <= 12 else 6.0) for i in range(1, 25)
    }
    history[date(2025, 9, 1)] = 20.0
    return first_target, series_groups, {group: history}


def test_hierarchical_predictions_conserve_group_total_and_use_as_of_trend():
    origin, series_groups, group_history = _case()
    forecasts = build_hierarchical_forecasts(
        series_groups,
        group_history,
        local_last12={(1, 10): 3.0, (2, 10): 1.0},
        pooled_last12={1: 10.0, 2: 0.0},
        target_months=(origin,),
    )
    assert tuple(forecasts) == FORECAST_METHODS
    seasonal = forecasts["seasonal_uniform"]
    assert sum(seasonal.values()) == pytest.approx(20.0)
    assert sum(forecasts["trend_uniform"].values()) == pytest.approx(40.0)
    assert sum(forecasts["trend_local"].values()) == pytest.approx(40.0)
    assert forecasts["trend_local"][(origin, 1, 10)] == pytest.approx(80 / 3)
    assert forecasts["trend_pooled"][(origin, 1, 10)] == pytest.approx(110 / 3)
    assert sum(forecasts["ma3_uniform"].values()) == pytest.approx(12.0)


def test_hierarchical_falls_back_when_prior_year_or_trend_missing():
    origin, series_groups, group_history = _case()
    group = (10, "motores")
    del group_history[group][date(2025, 9, 1)]
    forecasts = build_hierarchical_forecasts(
        series_groups, group_history, {}, {}, (origin,)
    )
    assert sum(forecasts["seasonal_uniform"].values()) == pytest.approx(12.0)
    assert sum(forecasts["trend_uniform"].values()) == pytest.approx(12.0)


def test_hierarchical_rejects_history_from_holdout():
    origin, series_groups, group_history = _case()
    group_history[(10, "motores")][origin] = 1000.0
    with pytest.raises(ValueError, match="Future history or invalid month"):
        build_hierarchical_forecasts(series_groups, group_history, {}, {}, (origin,))


def test_hierarchical_rejects_nonconsecutive_targets_and_negative_weights():
    origin, series_groups, group_history = _case()
    with pytest.raises(ValueError, match="consecutivo"):
        build_hierarchical_forecasts(
            series_groups,
            group_history,
            {},
            {},
            (origin, date(2026, 11, 1)),
        )
    with pytest.raises(ValueError, match="peso local"):
        build_hierarchical_forecasts(
            series_groups, group_history, {(1, 10): -1.0}, {}, (origin,)
        )


def test_metrics_use_same_sku_cohort_at_all_grains():
    origin, series_groups, _ = _case()
    actual = {(origin, 1, 10): 10.0, (origin, 2, 10): 0.0}
    predicted = {(origin, 1, 10): 0.0, (origin, 2, 10): 10.0}
    result = evaluate_forecasts(actual, predicted, series_groups)
    assert result["sku_location"]["wape"] == pytest.approx(200.0)
    assert result["category_location"]["wape"] == pytest.approx(0.0)
    assert result["location"]["wape"] == pytest.approx(0.0)
    assert result["total_month"]["wape"] == pytest.approx(0.0)
    with pytest.raises(ValueError, match="mesmas chaves"):
        evaluate_forecasts(actual, {(origin, 1, 10): 10.0}, series_groups)


def test_location_allocation_forecast_conserves_total_and_uses_asof_shares():
    origin = date(2026, 9, 1)
    histories = {
        (1, 10): {shift_months(origin, -offset): 2.0 for offset in range(1, 25)},
        (2, 10): {shift_months(origin, -offset): 1.0 for offset in range(1, 25)},
    }
    histories[(1, 10)][date(2025, 9, 1)] = 30.0
    histories[(2, 10)][date(2025, 9, 1)] = 10.0

    forecasts = build_location_allocation_forecasts(
        series_history=histories,
        target_months=(origin, shift_months(origin, 1)),
    )

    assert tuple(forecasts) == LOCATION_ALLOCATION_METHODS
    seasonal = forecasts["location_seasonal_share"]
    assert sum(seasonal[(origin, product, 10)] for product in (1, 2)) == pytest.approx(
        40.0
    )
    assert seasonal[(origin, 1, 10)] == pytest.approx(40.0 * (52 / 73 + 3 / 4) / 2)
    assert seasonal[(origin, 2, 10)] == pytest.approx(40.0 * (21 / 73 + 1 / 4) / 2)
    assert sum(
        forecasts["location_ma3_share"][(origin, product, 10)] for product in (1, 2)
    ) == pytest.approx(3.0)


def test_location_allocation_rejects_future_history_and_invalid_targets():
    origin = date(2026, 9, 1)
    histories = {(1, 10): {shift_months(origin, -1): 1.0}}
    histories[(1, 10)][origin] = 10.0
    with pytest.raises(ValueError, match="futuro"):
        build_location_allocation_forecasts(
            series_history=histories,
            target_months=(origin,),
        )
    del histories[(1, 10)][origin]
    with pytest.raises(ValueError, match="consecutivo"):
        build_location_allocation_forecasts(
            series_history=histories,
            target_months=(origin, date(2026, 11, 1)),
        )
