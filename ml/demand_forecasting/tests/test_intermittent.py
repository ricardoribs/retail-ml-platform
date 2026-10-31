"""Pure tests for Croston/SBA/TSB candidates."""

from __future__ import annotations

from datetime import date

import pytest

from ml.demand_forecasting.src.intermittent import (
    INTERMITTENT_METHODS,
    build_intermittent_forecasts,
    build_tsb_occurrence_probabilities,
    croston_forecast,
    sba_forecast,
    tsb_components,
    tsb_forecast,
    tsb_occurrence_probability,
)


def test_intermittent_estimators_keep_zero_for_series_without_sales():
    values = [0.0, 0.0, 0.0]
    assert croston_forecast(values) == 0.0
    assert sba_forecast(values) == 0.0
    assert tsb_forecast(values) == 0.0


def test_sba_corrects_croston_and_tsb_tracks_sale_probability():
    values = [0.0, 2.0, 0.0, 0.0, 4.0]
    assert croston_forecast(values) == pytest.approx(2.2 / 2.1)
    assert sba_forecast(values) == pytest.approx(0.95 * 2.2 / 2.1)
    assert tsb_forecast(values) == pytest.approx(0.1729 * 2.2)
    assert tsb_components(values) == pytest.approx((0.1729, 2.2))
    assert tsb_occurrence_probability(values) == pytest.approx(0.1729)


def test_forecasts_are_complete_and_do_not_use_future_history():
    origin = date(2026, 9, 1)
    history = {
        (10, 20): {
            date(2026, 6, 1): 2.0,
            date(2026, 8, 1): 4.0,
        }
    }
    result = build_intermittent_forecasts(
        series_history=history,
        target_months=(origin, date(2026, 10, 1), date(2026, 11, 1)),
    )
    expected_keys = {
        (date(2026, 9, 1), 10, 20),
        (date(2026, 10, 1), 10, 20),
        (date(2026, 11, 1), 10, 20),
    }
    assert set(result) == set(INTERMITTENT_METHODS)
    assert all(set(predictions) == expected_keys for predictions in result.values())
    assert all(
        predictions[(date(2026, 9, 1), 10, 20)]
        == predictions[(date(2026, 10, 1), 10, 20)]
        for predictions in result.values()
    )


def test_tsb_occurrence_probabilities_are_complete_and_frozen_in_horizon():
    origin = date(2026, 9, 1)
    history = {
        (10, 20): {
            date(2026, 6, 1): 2.0,
            date(2026, 8, 1): 4.0,
        }
    }
    probabilities = build_tsb_occurrence_probabilities(
        series_history=history,
        target_months=(origin, date(2026, 10, 1), date(2026, 11, 1)),
    )

    assert set(probabilities) == {
        (date(2026, 9, 1), 10, 20),
        (date(2026, 10, 1), 10, 20),
        (date(2026, 11, 1), 10, 20),
    }
    assert len(set(probabilities.values())) == 1
    assert 0 < probabilities[(date(2026, 9, 1), 10, 20)] < 1


def test_rejects_future_history_and_nonconsecutive_targets():
    with pytest.raises(ValueError, match="futuro"):
        build_intermittent_forecasts(
            series_history={(10, 20): {date(2026, 9, 1): 1.0}},
            target_months=(date(2026, 9, 1),),
        )
    with pytest.raises(ValueError, match="consecutivos"):
        build_intermittent_forecasts(
            series_history={(10, 20): {date(2026, 8, 1): 1.0}},
            target_months=(date(2026, 9, 1), date(2026, 11, 1)),
        )
