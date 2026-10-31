"""Pure tests for the hybrid forecast policy."""

from datetime import date

import pytest

from ml.demand_forecasting.src.segmentation import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    DEMAND_PATTERN_ERRATIC,
    DEMAND_PATTERN_INTERMITTENT,
    DEMAND_PATTERN_LUMPY,
    DEMAND_PATTERN_NO_DEMAND,
    DEMAND_PATTERN_SMOOTH,
    DEMAND_SEGMENT_INTERMITTENT,
    DEMAND_SEGMENT_NEW,
    DEMAND_SEGMENT_SEASONAL,
    DEMAND_SEGMENT_STABLE,
    STRATEGY_CATEGORY_LOCATION_ESTIMATE,
    STRATEGY_INTERMITTENT_REVIEW,
    STRATEGY_REGRESSION_CANDIDATE,
    STRATEGY_SEASONAL_BASELINE,
    classify_intermittent_pattern,
    classify_profile,
    classify_series,
)


def _history(values: list[float]) -> list[tuple[date, float]]:
    return [
        (date(2025, month, 1), value) for month, value in enumerate(values, start=1)
    ]


def test_classifies_recurrent_series_as_stable_regression_candidate():
    policy = classify_series(_history([2, 1, 3, 1, 4, 2, 3, 1, 2, 4, 2, 1]))

    assert policy.demand_segment == DEMAND_SEGMENT_STABLE
    assert policy.forecast_strategy == STRATEGY_REGRESSION_CANDIDATE
    assert policy.confidence_level == CONFIDENCE_HIGH
    assert policy.is_regression_candidate is True


def test_classifies_confirmed_seasonal_series_before_intermittent():
    policy = classify_series(
        _history([0, 4, 0, 6, 0, 5, 0, 3, 0, 6, 0, 4]),
        seasonal_signal_confirmed=True,
    )

    assert policy.demand_segment == DEMAND_SEGMENT_SEASONAL
    assert policy.forecast_strategy == STRATEGY_SEASONAL_BASELINE
    assert policy.confidence_level == CONFIDENCE_MEDIUM


def test_classifies_sparse_series_as_intermittent_review():
    policy = classify_series(_history([2, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0]))

    assert policy.demand_segment == DEMAND_SEGMENT_INTERMITTENT
    assert policy.forecast_strategy == STRATEGY_INTERMITTENT_REVIEW
    assert policy.confidence_level == CONFIDENCE_LOW
    assert policy.months_since_last_sale == 2


def test_does_not_apply_product_seasonality_to_locally_intermittent_series():
    policy = classify_series(
        _history([2, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0]),
        seasonal_signal_confirmed=True,
    )

    assert policy.demand_segment == DEMAND_SEGMENT_INTERMITTENT
    assert policy.forecast_strategy == STRATEGY_INTERMITTENT_REVIEW


def test_classifies_profile_without_collecting_full_history():
    policy = classify_profile(
        history_month_count=24,
        positive_demand_month_count=8,
        trailing_positive_month_count=3,
        months_since_last_sale=1,
        seasonal_signal_confirmed=True,
    )

    assert policy.demand_segment == DEMAND_SEGMENT_SEASONAL
    assert policy.forecast_strategy == STRATEGY_SEASONAL_BASELINE


def test_classifies_short_or_rare_series_as_category_location_estimate():
    policy = classify_series(_history([0, 0, 1, 0, 1]))

    assert policy.demand_segment == DEMAND_SEGMENT_NEW
    assert policy.forecast_strategy == STRATEGY_CATEGORY_LOCATION_ESTIMATE
    assert policy.confidence_level == CONFIDENCE_LOW


@pytest.mark.parametrize(
    ("values", "expected_pattern"),
    [
        ([2, 2, 2, 2], DEMAND_PATTERN_SMOOTH),
        ([1, 7, 1, 7], DEMAND_PATTERN_ERRATIC),
        ([2, 0, 2, 0], DEMAND_PATTERN_INTERMITTENT),
        ([1, 0, 7, 0], DEMAND_PATTERN_LUMPY),
    ],
)
def test_classifies_syntetos_boylan_patterns(values, expected_pattern):
    profile = classify_intermittent_pattern(_history(values))

    assert profile.demand_pattern == expected_pattern
    assert profile.adi is not None
    assert profile.cv_squared is not None


def test_classifies_series_without_sales_as_no_demand_pattern():
    profile = classify_intermittent_pattern(_history([0, 0, 0, 0]))

    assert profile.demand_pattern == DEMAND_PATTERN_NO_DEMAND
    assert profile.adi is None
    assert profile.cv_squared is None
    assert profile.positive_demand_month_count == 0


@pytest.mark.parametrize(
    "history",
    [
        [],
        [(date(2025, 2, 1), 1), (date(2025, 1, 1), 1)],
        [(date(2025, 1, 1), -1)],
    ],
)
def test_rejects_invalid_history(history):
    with pytest.raises(ValueError):
        classify_series(history)
