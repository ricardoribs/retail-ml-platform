"""Classifies demand series before selecting a forecast strategy.

Classification uses only history available through the cutoff. It does not
select a model based on realized values from a future horizon; selection
happens later, based on historical backtests by segment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from typing import Sequence

DEMAND_SEGMENT_STABLE = "stable"
DEMAND_SEGMENT_SEASONAL = "seasonal"
DEMAND_SEGMENT_INTERMITTENT = "intermittent"
DEMAND_SEGMENT_NEW = "new"

# Syntetos-Boylan classification. It is deliberately separate from the
# operational segments above: those route publication, while the patterns below
# explain how to evaluate intermittent demand in backtests.
DEMAND_PATTERN_SMOOTH = "smooth"
DEMAND_PATTERN_ERRATIC = "erratic"
DEMAND_PATTERN_INTERMITTENT = "intermittent"
DEMAND_PATTERN_LUMPY = "lumpy"
DEMAND_PATTERN_NO_DEMAND = "no_demand"

# Standard thresholds for the Syntetos-Boylan ADI/CV² classification. ADI measures
# the average interval between sales; CV² measures variation in positive
# quantities. The boundary is documented so it is not selected using holdout actuals.
SYNTETOS_BOYLAN_ADI_THRESHOLD = 1.32
SYNTETOS_BOYLAN_CV2_THRESHOLD = 0.49

STRATEGY_REGRESSION_CANDIDATE = "regression_candidate"
STRATEGY_SEASONAL_BASELINE = "baseline_seasonal"
STRATEGY_INTERMITTENT_REVIEW = "intermittent_review"
STRATEGY_CATEGORY_LOCATION_ESTIMATE = "category_location_estimate"

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"

MIN_HISTORY_MONTHS = 6
MIN_POSITIVE_MONTHS = 3
STABLE_TRAILING_MONTHS = 12
MIN_STABLE_POSITIVE_MONTHS = 9
MIN_SEASONAL_HISTORY_MONTHS = 12
MIN_SEASONAL_POSITIVE_MONTHS = 6
MIN_SEASONAL_TRAILING_POSITIVE_MONTHS = 3


@dataclass(frozen=True)
class SeriesPolicy:
    """Initial profile and strategy for a product-location series."""

    demand_segment: str
    forecast_strategy: str
    confidence_level: str
    selection_reason: str
    history_month_count: int
    positive_demand_month_count: int
    trailing_positive_month_count: int
    months_since_last_sale: int | None

    @property
    def is_regression_candidate(self) -> bool:
        """Indicate whether the series can use regression after segment validation."""
        return self.demand_segment == DEMAND_SEGMENT_STABLE


@dataclass(frozen=True)
class IntermittentDemandProfile:
    """ADI/CV² profile calculated only from demand available at the origin.

    ``cv_squared`` is ``None`` when there have been no sales because variation
    in positive demand size is undefined. The ``no_demand`` classification
    prevents assigning a demand policy to a series without positive observations.
    """

    demand_pattern: str
    adi: float | None
    cv_squared: float | None
    history_month_count: int
    positive_demand_month_count: int


def _validate_history(
    history: Sequence[tuple[date, float]],
) -> list[tuple[date, float]]:
    """Validate a dense, ordered monthly series without invalid values."""
    if not history:
        raise ValueError("Series history cannot be empty")

    normalized = [(month, float(quantity)) for month, quantity in history]
    months = [month for month, _ in normalized]
    if months != sorted(set(months)):
        raise ValueError("History must contain distinct, increasing months")
    if any(not math.isfinite(quantity) or quantity < 0 for _, quantity in normalized):
        raise ValueError("History must contain only finite, non-negative demand")
    return normalized


def classify_intermittent_pattern(
    history: Sequence[tuple[date, float]],
) -> IntermittentDemandProfile:
    """Classify a dense monthly series by ADI and CV².

    The implementation calculates ADI across all months and CV² only across
    positive demand sizes. It uses population variance because the full history
    available at the origin, rather than a sample from an observed process, is
    routed in the backtest.

    The function does not recommend a forecast or read future months. It is used
    to stratify metrics and decide whether a series merits new modeling or a
    review/replenishment policy.
    """
    normalized = _validate_history(history)
    positive_values = [quantity for _, quantity in normalized if quantity > 0]
    history_month_count = len(normalized)
    positive_demand_month_count = len(positive_values)

    if not positive_values:
        return IntermittentDemandProfile(
            demand_pattern=DEMAND_PATTERN_NO_DEMAND,
            adi=None,
            cv_squared=None,
            history_month_count=history_month_count,
            positive_demand_month_count=0,
        )

    adi = history_month_count / positive_demand_month_count
    positive_mean = sum(positive_values) / positive_demand_month_count
    cv_squared = (
        sum((value - positive_mean) ** 2 for value in positive_values)
        / positive_demand_month_count
        / positive_mean**2
    )

    if adi < SYNTETOS_BOYLAN_ADI_THRESHOLD:
        demand_pattern = (
            DEMAND_PATTERN_SMOOTH
            if cv_squared < SYNTETOS_BOYLAN_CV2_THRESHOLD
            else DEMAND_PATTERN_ERRATIC
        )
    else:
        demand_pattern = (
            DEMAND_PATTERN_INTERMITTENT
            if cv_squared < SYNTETOS_BOYLAN_CV2_THRESHOLD
            else DEMAND_PATTERN_LUMPY
        )

    return IntermittentDemandProfile(
        demand_pattern=demand_pattern,
        adi=adi,
        cv_squared=cv_squared,
        history_month_count=history_month_count,
        positive_demand_month_count=positive_demand_month_count,
    )


def classify_series(
    history: Sequence[tuple[date, float]],
    *,
    seasonal_signal_confirmed: bool = False,
) -> SeriesPolicy:
    """Classify the series using only data from before the forecast horizon.

    ``seasonal_signal_confirmed`` must come from historical product seasonality
    validation; this function does not infer seasonality from random variation
    in a short series.
    """
    normalized = _validate_history(history)
    values = [quantity for _, quantity in normalized]
    history_month_count = len(values)
    positive_demand_month_count = sum(quantity > 0 for quantity in values)
    trailing_values = values[-STABLE_TRAILING_MONTHS:]
    trailing_positive_month_count = sum(quantity > 0 for quantity in trailing_values)

    positive_positions = [
        index for index, quantity in enumerate(values) if quantity > 0
    ]
    months_since_last_sale = (
        history_month_count - 1 - positive_positions[-1] if positive_positions else None
    )

    return classify_profile(
        history_month_count=history_month_count,
        positive_demand_month_count=positive_demand_month_count,
        trailing_positive_month_count=trailing_positive_month_count,
        months_since_last_sale=months_since_last_sale,
        seasonal_signal_confirmed=seasonal_signal_confirmed,
    )


def classify_profile(
    *,
    history_month_count: int,
    positive_demand_month_count: int,
    trailing_positive_month_count: int,
    months_since_last_sale: int | None,
    seasonal_signal_confirmed: bool = False,
) -> SeriesPolicy:
    """Classify a series summary calculated through the cutoff date.

    This is the counterpart to :func:`classify_series` for Spark jobs: the
    notebook aggregates the series at its source grain and calls this function
    once per product-location pair. The seasonality signal is product-level
    context; the seasonal route still requires observable recurrence at the location.
    """
    counts = (
        history_month_count,
        positive_demand_month_count,
        trailing_positive_month_count,
    )
    if any(not isinstance(count, int) or count < 0 for count in counts):
        raise ValueError("Profile counts must be non-negative integers")
    if positive_demand_month_count > history_month_count:
        raise ValueError("Positive months cannot exceed the history length")
    if trailing_positive_month_count > min(history_month_count, STABLE_TRAILING_MONTHS):
        raise ValueError("Recent positive months exceed the history window")
    if months_since_last_sale is not None and (
        not isinstance(months_since_last_sale, int)
        or months_since_last_sale < 0
        or months_since_last_sale >= history_month_count
    ):
        raise ValueError("months_since_last_sale is invalid for the history")
    if history_month_count == 0:
        raise ValueError("Series history cannot be empty")

    policy_fields = {
        "history_month_count": history_month_count,
        "positive_demand_month_count": positive_demand_month_count,
        "trailing_positive_month_count": trailing_positive_month_count,
        "months_since_last_sale": months_since_last_sale,
    }

    if (
        history_month_count < MIN_HISTORY_MONTHS
        or positive_demand_month_count < MIN_POSITIVE_MONTHS
    ):
        return SeriesPolicy(
            demand_segment=DEMAND_SEGMENT_NEW,
            forecast_strategy=STRATEGY_CATEGORY_LOCATION_ESTIMATE,
            confidence_level=CONFIDENCE_LOW,
            selection_reason=(
                "Insufficient history to estimate series demand; use the category and "
                "location reference as a planning estimate."
            ),
            **policy_fields,
        )

    if (
        history_month_count >= STABLE_TRAILING_MONTHS
        and trailing_positive_month_count >= MIN_STABLE_POSITIVE_MONTHS
    ):
        return SeriesPolicy(
            demand_segment=DEMAND_SEGMENT_STABLE,
            forecast_strategy=STRATEGY_REGRESSION_CANDIDATE,
            confidence_level=CONFIDENCE_HIGH,
            selection_reason=(
                "Recurring series in the last 12 months; use regression only if it "
                "beats the benchmark in the segment backtest."
            ),
            **policy_fields,
        )

    if (
        seasonal_signal_confirmed
        and history_month_count >= MIN_SEASONAL_HISTORY_MONTHS
        and positive_demand_month_count >= MIN_SEASONAL_POSITIVE_MONTHS
        and trailing_positive_month_count >= MIN_SEASONAL_TRAILING_POSITIVE_MONTHS
    ):
        return SeriesPolicy(
            demand_segment=DEMAND_SEGMENT_SEASONAL,
            forecast_strategy=STRATEGY_SEASONAL_BASELINE,
            confidence_level=CONFIDENCE_MEDIUM,
            selection_reason=(
                "Product has a validated seasonal signal; use the same month from "
                "the previous year with a documented fallback."
            ),
            **policy_fields,
        )

    return SeriesPolicy(
        demand_segment=DEMAND_SEGMENT_INTERMITTENT,
        forecast_strategy=STRATEGY_INTERMITTENT_REVIEW,
        confidence_level=CONFIDENCE_LOW,
        selection_reason=(
            "Sparse or irregular sales; publish the estimate for planning only and "
            "require replenishment review."
        ),
        **policy_fields,
    )
