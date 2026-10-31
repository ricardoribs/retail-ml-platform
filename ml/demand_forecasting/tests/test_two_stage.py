"""Pure contract tests for the experimental candidate, without local Spark."""

from __future__ import annotations

import math

import pytest

from ml.demand_forecasting.src.two_stage import (
    EXPECTED_MODEL_NAME,
    MODEL_NAME,
    THRESHOLDS,
    calculate_occurrence_metrics,
    expected_quantity,
    gated_quantity,
)


def test_two_stage_exposes_predefined_thresholds():
    assert MODEL_NAME == "two_stage_gbt"
    assert EXPECTED_MODEL_NAME == "two_stage_gbt_expected"
    assert THRESHOLDS == (0.1, 0.25, 0.5, 0.75)


def test_occurrence_metrics_reward_perfect_probability_ranking_and_calibration():
    metrics = calculate_occurrence_metrics(
        actual_quantity=[0.0, 4.0, 0.0, 2.0],
        occurrence_probability=[0.0, 1.0, 0.0, 1.0],
        reference_probability=0.5,
    )

    assert metrics["event_rate"] == 0.5
    assert metrics["brier_score"] == 0.0
    assert metrics["brier_skill_score"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["average_precision"] == 1.0
    assert metrics["decision_precision"] == 1.0
    assert metrics["decision_recall"] == 1.0


def test_occurrence_metrics_reject_invalid_evaluation_inputs():
    with pytest.raises(ValueError, match="same size"):
        calculate_occurrence_metrics([0.0], [0.2, 0.3], 0.2)
    with pytest.raises(ValueError, match="reference probability"):
        calculate_occurrence_metrics([0.0, 1.0], [0.2, 0.3], 1.1)
    with pytest.raises(ValueError, match="with and without sales"):
        calculate_occurrence_metrics([0.0, 0.0], [0.2, 0.3], 0.2)


def test_occurrence_metrics_use_the_requested_decision_threshold():
    metrics = calculate_occurrence_metrics(
        actual_quantity=[0.0, 1.0, 0.0, 1.0],
        occurrence_probability=[0.10, 0.20, 0.10, 0.20],
        reference_probability=0.5,
        decision_threshold=0.15,
    )

    assert metrics["decision_positive_rate"] == 0.5
    assert metrics["decision_precision"] == 1.0
    assert metrics["decision_recall"] == 1.0


@pytest.mark.parametrize(
    ("probability", "amount", "threshold", "expected"),
    [
        (0.49, 8.0, 0.5, 0.0),
        (0.50, 8.0, 0.5, 8.0),
        (0.80, -2.0, 0.5, 0.0),
        (0.75, 3.5, 0.75, 3.5),
    ],
)
def test_gate_applies_occurrence_before_positive_amount(
    probability: float, amount: float, threshold: float, expected: float
):
    assert gated_quantity(probability, amount, threshold) == expected


@pytest.mark.parametrize("bad_probability", [-0.01, 1.01, math.nan, math.inf])
def test_gate_rejects_invalid_probability(bad_probability: float):
    with pytest.raises(ValueError, match="Probabilidade"):
        gated_quantity(bad_probability, 2.0, 0.5)


@pytest.mark.parametrize("bad_amount", [math.nan, math.inf, -math.inf])
def test_gate_rejects_invalid_positive_amount(bad_amount: float):
    with pytest.raises(ValueError, match="Quantidade"):
        gated_quantity(0.8, bad_amount, 0.5)


def test_expected_quantity_preserves_occurrence_uncertainty():
    assert expected_quantity(0.25, 8.0) == 2.0
    assert expected_quantity(0.8, -2.0) == 0.0


@pytest.mark.parametrize("bad_probability", [-0.01, 1.01, math.nan, math.inf])
def test_expected_quantity_rejects_invalid_probability(bad_probability: float):
    with pytest.raises(ValueError, match="Probabilidade"):
        expected_quantity(bad_probability, 2.0)


@pytest.mark.parametrize("bad_threshold", [0.0, 1.0, math.nan, math.inf])
def test_gate_rejects_invalid_threshold(bad_threshold: float):
    with pytest.raises(ValueError, match="limiar"):
        gated_quantity(0.8, 2.0, bad_threshold)
