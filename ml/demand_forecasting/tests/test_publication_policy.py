"""Tests for choosing between a published forecast and reviewed planning."""

from datetime import date

from ml.demand_forecasting.src.publication_policy import (
    MODEL_BASELINE_SEASONAL,
    MODEL_BASELINE_ZERO,
    STATUS_APPROVED_FORECAST,
    STATUS_PLANNING_REVIEW,
    decide_publication,
    decide_validated_series_model,
    seasonal_baseline_is_approved,
)
from ml.demand_forecasting.src.segmentation import classify_series


def _history(values: list[float]) -> list[tuple[date, float]]:
    return [
        (date(2025, month, 1), value) for month, value in enumerate(values, start=1)
    ]


def test_seasonal_baseline_requires_lower_segment_wape_than_zero():
    assert seasonal_baseline_is_approved(
        [
            {"model_name": MODEL_BASELINE_ZERO, "wape": 100.0},
            {"model_name": MODEL_BASELINE_SEASONAL, "wape": 99.9},
        ]
    )
    assert not seasonal_baseline_is_approved(
        [
            {"model_name": MODEL_BASELINE_ZERO, "wape": 100.0},
            {"model_name": MODEL_BASELINE_SEASONAL, "wape": 100.0},
        ]
    )


def test_rejected_seasonal_candidate_becomes_planning_review():
    policy = classify_series(
        _history([0, 4, 0, 6, 0, 5, 0, 3, 0, 6, 0, 4]),
        seasonal_signal_confirmed=True,
    )

    decision = decide_publication(policy, seasonal_baseline_approved=False)

    assert decision.applied_model_name == MODEL_BASELINE_ZERO
    assert decision.publication_status == STATUS_PLANNING_REVIEW
    assert decision.review_required is True
    assert decision.automation_allowed is False


def test_approved_seasonal_baseline_remains_planning_input_not_auto_purchase():
    policy = classify_series(
        _history([0, 4, 0, 6, 0, 5, 0, 3, 0, 6, 0, 4]),
        seasonal_signal_confirmed=True,
    )

    decision = decide_publication(policy, seasonal_baseline_approved=True)

    assert decision.applied_model_name == MODEL_BASELINE_SEASONAL
    assert decision.publication_status == STATUS_APPROVED_FORECAST
    assert decision.automation_allowed is False


def test_intermittent_series_requires_monthly_review():
    policy = classify_series(_history([2, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0]))

    decision = decide_publication(policy, seasonal_baseline_approved=False)

    assert decision.applied_model_name == MODEL_BASELINE_ZERO
    assert decision.review_cadence == "monthly"
    assert decision.publication_status == STATUS_PLANNING_REVIEW


def test_validated_series_champion_is_an_approved_forecast_not_auto_purchase():
    policy = classify_series(_history([1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2]))

    decision = decide_validated_series_model(
        policy,
        selected_model_name="baseline_ma3",
        validation_status="validated_champion",
        selection_reason="baseline_ma3 venceu em 2 de 3 folds",
    )

    assert decision.applied_model_name == "baseline_ma3"
    assert decision.publication_status == STATUS_APPROVED_FORECAST
    assert decision.automation_allowed is False


def test_series_without_enough_signal_stays_in_review():
    policy = classify_series(_history([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

    decision = decide_validated_series_model(
        policy,
        selected_model_name=MODEL_BASELINE_ZERO,
        validation_status="insufficient_validation_signal",
        selection_reason="sinal insuficiente",
    )

    assert decision.publication_status == STATUS_PLANNING_REVIEW


def test_validated_sparse_zero_is_published_but_keeps_monthly_review():
    policy = classify_series(_history([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]))

    decision = decide_validated_series_model(
        policy,
        selected_model_name=MODEL_BASELINE_ZERO,
        validation_status="validated_zero_sparse",
        selection_reason="zero venceu os candidatos com uma ocorrencia",
    )

    assert decision.publication_status == STATUS_APPROVED_FORECAST
    assert decision.review_required is True
    assert decision.automation_allowed is False
