"""Pure contracts for the as-of candidate; no test writes to Databricks."""

from __future__ import annotations

from datetime import date

import pytest

from ml.demand_forecasting.src.asof_allocation import (
    build_asof_forecasts,
    history_segment,
)
from ml.demand_forecasting.src.modeling import shift_months


def _case():
    origin = date(2026, 9, 1)
    series_groups = {(1, 10): (10, "motor"), (2, 10): (10, "motor")}
    series_history = {
        (1, 10): {shift_months(origin, -i): 2.0 for i in range(1, 25)},
        (2, 10): {shift_months(origin, -i): 0.0 for i in range(1, 25)},
    }
    series_history[(2, 10)][date(2025, 9, 1)] = 2.0
    group_history = {
        (10, "motor"): {
            month: sum(history[month] for history in series_history.values())
            for month in series_history[(1, 10)]
        }
    }
    return origin, series_groups, series_history, group_history


def test_adaptive_uses_target_month_seasonality_and_conserves_group_total():
    origin, groups, history, group_history = _case()
    predictions = build_asof_forecasts(
        series_groups=groups,
        series_history=history,
        group_history=group_history,
        target_months=(origin, shift_months(origin, 1)),
    )
    assert predictions["zero"][(origin, 1, 10)] == 0
    assert predictions["ma3_frozen"][(origin, 1, 10)] == 2
    assert predictions["seasonal_series"][(origin, 2, 10)] == 2
    september = predictions["seasonal_adaptive"]
    october = predictions["seasonal_adaptive"]
    assert september[(origin, 1, 10)] == pytest.approx(4 * (24 / 26 + 1 / 2) / 2)
    assert september[(origin, 2, 10)] == pytest.approx(4 * (2 / 26 + 1 / 2) / 2)
    assert sum(september[(origin, product, 10)] for product in (1, 2)) == 4
    assert october[(shift_months(origin, 1), 2, 10)] == pytest.approx(2 / 26)


def test_history_segments_use_only_pre_origin_months():
    origin, _, history, _ = _case()
    assert history_segment(history[(1, 10)], origin) == "recorrente_3_6"
    assert history_segment(history[(2, 10)], origin) == "sem_venda_6_6"
    partial = {shift_months(origin, -1): 1.0}
    assert history_segment(partial, origin) == "historico_incompleto"
    partial[origin] = 1000.0
    with pytest.raises(ValueError, match="futuro"):
        history_segment(partial, origin)


def test_adaptive_rejects_future_history_and_missing_cohort():
    origin, groups, history, group_history = _case()
    history[(1, 10)][origin] = 1000.0
    with pytest.raises(ValueError, match="futuro"):
        build_asof_forecasts(
            series_groups=groups,
            series_history=history,
            group_history=group_history,
            target_months=(origin,),
        )
    del history[(1, 10)][origin]
    del history[(2, 10)]
    with pytest.raises(ValueError, match="cohort"):
        build_asof_forecasts(
            series_groups=groups,
            series_history=history,
            group_history=group_history,
            target_months=(origin,),
        )


def test_adaptive_rejects_inconsistent_group_totals():
    origin, groups, history, group_history = _case()
    group_history[(10, "motor")][date(2025, 9, 1)] += 1
    with pytest.raises(ValueError, match="reconciliam"):
        build_asof_forecasts(
            series_groups=groups,
            series_history=history,
            group_history=group_history,
            target_months=(origin,),
        )
