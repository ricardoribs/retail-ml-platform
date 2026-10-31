"""Pure tests for src/seasonality.py, without Spark.


Cover the exact official threshold boundaries: 1.20 and 0.80 for the
seasonal index, plus minimum observation counts of 3 and 2.
"""

from __future__ import annotations

from ml.demand_forecasting.src.seasonality import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    HIGH_CONFIDENCE_MIN_OBSERVATIONS,
    MEDIUM_CONFIDENCE_MIN_OBSERVATIONS,
    SEASONAL_LOW_INDEX_THRESHOLD,
    SEASONAL_PEAK_INDEX_THRESHOLD,
    SEASONALITY_CLASS_LOW,
    SEASONALITY_CLASS_NEUTRAL,
    SEASONALITY_CLASS_PEAK,
    SEASONALITY_MIN_HISTORY_MONTHS,
    SEASONALITY_MIN_POSITIVE_MONTHS,
    classify_confidence,
    classify_seasonality_class,
)


def test_eligibility_thresholds_match_decided_rule():
    assert SEASONALITY_MIN_HISTORY_MONTHS == 12
    assert SEASONALITY_MIN_POSITIVE_MONTHS == 6


def test_seasonality_index_thresholds_match_decided_rule():
    assert SEASONAL_PEAK_INDEX_THRESHOLD == 1.20
    assert SEASONAL_LOW_INDEX_THRESHOLD == 0.80


def test_confidence_thresholds_match_decided_rule():
    assert HIGH_CONFIDENCE_MIN_OBSERVATIONS == 3
    assert MEDIUM_CONFIDENCE_MIN_OBSERVATIONS == 2


def test_classify_seasonality_class_peak_boundary():
    assert classify_seasonality_class(1.20) == SEASONALITY_CLASS_PEAK
    assert classify_seasonality_class(1.21) == SEASONALITY_CLASS_PEAK


def test_classify_seasonality_class_just_below_peak_is_neutral():
    assert classify_seasonality_class(1.19) == SEASONALITY_CLASS_NEUTRAL


def test_classify_seasonality_class_low_boundary():
    assert classify_seasonality_class(0.80) == SEASONALITY_CLASS_LOW
    assert classify_seasonality_class(0.79) == SEASONALITY_CLASS_LOW


def test_classify_seasonality_class_just_above_low_is_neutral():
    assert classify_seasonality_class(0.81) == SEASONALITY_CLASS_NEUTRAL


def test_classify_seasonality_class_neutral_midpoint():
    assert classify_seasonality_class(1.0) == SEASONALITY_CLASS_NEUTRAL


def test_classify_confidence_high_boundary():
    assert classify_confidence(3) == CONFIDENCE_HIGH
    assert classify_confidence(4) == CONFIDENCE_HIGH


def test_classify_confidence_medium_boundary():
    assert classify_confidence(2) == CONFIDENCE_MEDIUM


def test_classify_confidence_low_boundary():
    assert classify_confidence(1) == CONFIDENCE_LOW
    assert classify_confidence(0) == CONFIDENCE_LOW
