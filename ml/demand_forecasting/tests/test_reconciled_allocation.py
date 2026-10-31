"""Pure contracts for the reconciled SKU-location candidate."""

from __future__ import annotations

from datetime import date

import pytest

from ml.demand_forecasting.src.modeling import shift_months
from ml.demand_forecasting.src.reconciled_allocation import (
    build_reconciled_allocation_forecasts,
    reconcile_product_location_matrix,
)


def test_ipf_reconciles_product_and_location_margins():
    values = reconcile_product_location_matrix(
        initial_weights={(1, 10): 8.0, (1, 20): 2.0, (2, 10): 1.0, (2, 20): 9.0},
        product_totals={1: 30.0, 2: 70.0},
        location_totals={10: 40.0, 20: 60.0},
    )
    assert sum(values[(1, location)] for location in (10, 20)) == pytest.approx(30.0)
    assert sum(values[(2, location)] for location in (10, 20)) == pytest.approx(70.0)
    assert sum(values[(product, 10)] for product in (1, 2)) == pytest.approx(40.0)
    assert sum(values[(product, 20)] for product in (1, 2)) == pytest.approx(60.0)


def test_ipf_rejects_incompatible_margins():
    with pytest.raises(ValueError, match="mesma massa"):
        reconcile_product_location_matrix(
            initial_weights={(1, 10): 1.0},
            product_totals={1: 2.0},
            location_totals={10: 1.0},
        )


def test_reconciled_forecast_conserves_both_seasonal_margins():
    origin = date(2026, 9, 1)
    groups = {
        (1, 10): (10, "motor"),
        (1, 20): (20, "motor"),
        (2, 10): (10, "motor"),
        (2, 20): (20, "motor"),
    }
    history = {
        series: {shift_months(origin, -offset): 1.0 for offset in range(1, 25)}
        for series in groups
    }
    history[(1, 10)][date(2025, 9, 1)] = 6.0
    history[(1, 20)][date(2025, 9, 1)] = 9.0
    history[(2, 10)][date(2025, 9, 1)] = 4.0
    history[(2, 20)][date(2025, 9, 1)] = 11.0

    forecast = build_reconciled_allocation_forecasts(
        series_groups=groups,
        series_history=history,
        target_months=(origin,),
    )

    assert sum(
        forecast[(origin, 1, location)] for location in (10, 20)
    ) == pytest.approx(15.0)
    assert sum(
        forecast[(origin, 2, location)] for location in (10, 20)
    ) == pytest.approx(15.0)
    assert sum(forecast[(origin, product, 10)] for product in (1, 2)) == pytest.approx(
        10.0
    )
    assert sum(forecast[(origin, product, 20)] for product in (1, 2)) == pytest.approx(
        20.0
    )
    assert all(value >= 0 for value in forecast.values())


def test_reconciled_forecast_assigns_nonzero_weight_to_known_zero_cell():
    origin = date(2026, 9, 1)
    groups = {
        (1, 10): (10, "motor"),
        (1, 20): (20, "motor"),
        (2, 10): (10, "motor"),
        (2, 20): (20, "motor"),
    }
    history = {
        (1, 10): {shift_months(origin, -offset): 3.0 for offset in range(1, 25)},
        (1, 20): {shift_months(origin, -offset): 1.0 for offset in range(1, 25)},
        (2, 10): {shift_months(origin, -offset): 1.0 for offset in range(1, 25)},
        (2, 20): {shift_months(origin, -offset): 0.0 for offset in range(1, 25)},
    }
    forecast = build_reconciled_allocation_forecasts(
        series_groups=groups,
        series_history=history,
        target_months=(origin,),
    )
    assert forecast[(origin, 2, 20)] > 0


def test_reconciled_forecast_rejects_future_history():
    origin = date(2026, 9, 1)
    groups = {(1, 10): (10, "motor")}
    history = {(1, 10): {origin: 1.0}}
    with pytest.raises(ValueError, match="futuro"):
        build_reconciled_allocation_forecasts(
            series_groups=groups,
            series_history=history,
            target_months=(origin,),
        )
