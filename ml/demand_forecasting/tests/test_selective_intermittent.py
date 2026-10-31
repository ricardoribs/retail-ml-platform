"""Testes puros do forecast seletivo calibrado em janelas passadas."""

from __future__ import annotations

from datetime import date

import pytest

from ml.demand_forecasting.src.selective_intermittent import (
    build_conditional_quantity_forecast,
    build_occurrence_probabilities,
    build_selective_candidates,
    select_threshold_from_history,
    threshold_predictions,
)


def test_threshold_suppresses_low_probability_without_changing_high_probability():
    first = (date(2026, 9, 1), 1, 10)
    second = (date(2026, 9, 1), 2, 10)
    predicted = threshold_predictions(
        expected_demand={first: 5.0, second: 7.0},
        probabilities={first: 0.8, second: 0.2},
        threshold=0.5,
    )
    assert predicted == {first: 5.0, second: 0.0}


def test_inner_history_selection_prefers_threshold_that_reduces_error():
    first = (date(2026, 9, 1), 1, 10)
    second = (date(2026, 9, 1), 2, 10)
    selected = select_threshold_from_history(
        actual={first: 5.0, second: 0.0},
        probabilities={first: 0.8, second: 0.2},
        thresholds=(0.2, 0.5, 0.9),
    )
    assert selected.threshold == 0.5
    assert selected.precision == 1.0
    assert selected.recall == 1.0
    assert selected.fbeta == 1.0
    assert selected.predicted_positive_rows == 1


def test_candidates_are_complete_and_reject_future_history():
    origin = date(2026, 9, 1)
    groups = {(1, 10): (10, "motor")}
    history = {(1, 10): {date(2026, 8, 1): 3.0}}
    candidates = build_selective_candidates(
        series_groups=groups,
        series_history=history,
        target_months=(origin, date(2026, 10, 1)),
        thresholds=(0.5, 0.9),
    )
    expected_keys = {(origin, 1, 10), (date(2026, 10, 1), 1, 10)}
    assert all(set(candidate) == expected_keys for candidate in candidates.values())

    history[(1, 10)][origin] = 1.0
    with pytest.raises(ValueError, match="futuro"):
        build_selective_candidates(
            series_groups=groups,
            series_history=history,
            target_months=(origin,),
        )


def test_selective_candidate_publishes_conditional_quantity_after_sale_decision():
    origin = date(2026, 9, 1)
    groups = {(1, 10): (10, "motor")}
    history = {
        (1, 10): {
            date(2025, 9, 1): 0.0,
            date(2026, 7, 1): 0.0,
            date(2026, 8, 1): 4.0,
        }
    }
    probabilities = build_occurrence_probabilities(
        series_groups=groups,
        series_history=history,
        target_months=(origin,),
    )
    conditional = build_conditional_quantity_forecast(
        series_groups=groups,
        series_history=history,
        target_months=(origin,),
        probabilities=probabilities,
    )
    candidate = build_selective_candidates(
        series_groups=groups,
        series_history=history,
        target_months=(origin,),
        thresholds=(0.0,),
    )[0.0]
    key = (origin, 1, 10)
    assert 0 < probabilities[key] < 1
    assert candidate[key] == pytest.approx(conditional[key])
    assert candidate[key] > probabilities[key] * conditional[key]


def test_threshold_requires_complete_matching_keys():
    first = (date(2026, 9, 1), 1, 10)
    with pytest.raises(ValueError, match="mesmas chaves"):
        threshold_predictions(
            expected_demand={first: 1.0},
            probabilities={},
            threshold=0.5,
        )


def test_threshold_above_one_is_not_a_valid_zero_fallback():
    first = (date(2026, 9, 1), 1, 10)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        threshold_predictions(
            expected_demand={first: 1.0},
            probabilities={first: 0.5},
            threshold=1.01,
        )


def test_f2_selection_favors_recall_without_using_forecast_magnitude():
    keys = tuple((date(2026, 9, 1), product, 10) for product in range(1, 5))
    selected = select_threshold_from_history(
        actual={keys[0]: 20.0, keys[1]: 1.0, keys[2]: 0.0, keys[3]: 0.0},
        probabilities={keys[0]: 0.9, keys[1]: 0.4, keys[2]: 0.3, keys[3]: 0.1},
        thresholds=(0.3, 0.5),
        beta=2.0,
    )
    assert selected.threshold == 0.3
    assert selected.true_positive_rows == 2
    assert selected.false_positive_rows == 1
    assert selected.false_negative_rows == 0


def test_selection_rejects_invalid_threshold_grid():
    first = (date(2026, 9, 1), 1, 10)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        select_threshold_from_history(
            actual={first: 1.0},
            probabilities={first: 0.5},
            thresholds=(1.01,),
        )
