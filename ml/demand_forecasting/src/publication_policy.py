"""Publication decisions for the hybrid forecast policy.

Demand profiling and backtest evidence are separate stages. For example, a
series with seasonal context does not automatically receive a seasonal
forecast: the candidate must outperform zero in the segment's recursive
backtest."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ml.demand_forecasting.src.segmentation import (
    DEMAND_SEGMENT_INTERMITTENT,
    DEMAND_SEGMENT_NEW,
    DEMAND_SEGMENT_SEASONAL,
    DEMAND_SEGMENT_STABLE,
    SeriesPolicy,
)

MODEL_BASELINE_SEASONAL = "baseline_seasonal"
MODEL_BASELINE_ZERO = "baseline_zero"

STATUS_APPROVED_FORECAST = "approved_forecast"
STATUS_PLANNING_REVIEW = "planning_review"

STRATEGY_SEASONAL_BASELINE = "seasonal_baseline"
STRATEGY_ZERO_REVIEW = "baseline_zero_review"
STRATEGY_CATEGORY_LOCATION_REVIEW = "category_location_review"
STRATEGY_VALIDATED_SERIES_MODEL = "validated_series_model"


@dataclass(frozen=True)
class PublicationDecision:
    """Applied strategy, separated from the profile candidate strategy."""

    applied_model_name: str
    applied_strategy: str
    publication_status: str
    review_required: bool
    automation_allowed: bool
    review_cadence: str
    decision_reason: str


def _metric_by_model(
    metric_rows: Sequence[Mapping[str, Any]], model_name: str
) -> Mapping[str, Any] | None:
    matching = [row for row in metric_rows if row.get("model_name") == model_name]
    if len(matching) > 1:
        raise ValueError(f"Duplicate metrics for {model_name}")
    return matching[0] if matching else None


def seasonal_baseline_is_approved(
    metric_rows: Sequence[Mapping[str, Any]],
) -> bool:
    """Approve the seasonal baseline only if it lowers WAPE versus zero in the same segment."""
    seasonal = _metric_by_model(metric_rows, MODEL_BASELINE_SEASONAL)
    zero = _metric_by_model(metric_rows, MODEL_BASELINE_ZERO)
    if seasonal is None or zero is None:
        return False

    seasonal_wape = seasonal.get("wape")
    zero_wape = zero.get("wape")
    if seasonal_wape is None or zero_wape is None:
        return False
    try:
        seasonal_value = float(seasonal_wape)
        zero_value = float(zero_wape)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(seasonal_value)
        and math.isfinite(zero_value)
        and seasonal_value < zero_value
    )


def decide_publication(
    policy: SeriesPolicy,
    *,
    seasonal_baseline_approved: bool,
    regression_segment_approved: bool = False,
) -> PublicationDecision:
    """Define the applied route without allowing automatic replenishment without required inputs.

    Lead time, available inventory, service level, and stockout cost are not part
    of the modeling data. Therefore, even an approved forecast remains a planning
    input and does not become an automatic purchase order.
    """
    if policy.demand_segment == DEMAND_SEGMENT_SEASONAL:
        if seasonal_baseline_approved:
            return PublicationDecision(
                applied_model_name=MODEL_BASELINE_SEASONAL,
                applied_strategy=STRATEGY_SEASONAL_BASELINE,
                publication_status=STATUS_APPROVED_FORECAST,
                review_required=False,
                automation_allowed=False,
                review_cadence="monthly",
                decision_reason=(
                    "Seasonal baseline beat zero in the segment's recursive backtest; "
                    "use it as an input for monthly planning."
                ),
            )
        return _zero_review_decision(
            "Aggregate seasonality did not beat zero in the segment backtest; "
            "keep it as context for review only."
        )

    if policy.demand_segment == DEMAND_SEGMENT_STABLE:
        if regression_segment_approved:
            raise NotImplementedError(
                "The regression route requires segmented metrics and a versioned model"
            )
        return _zero_review_decision(
            "No regression is approved for the recurring series in this segment."
        )

    if policy.demand_segment == DEMAND_SEGMENT_NEW:
        return PublicationDecision(
            applied_model_name=MODEL_BASELINE_ZERO,
            applied_strategy=STRATEGY_CATEGORY_LOCATION_REVIEW,
            publication_status=STATUS_PLANNING_REVIEW,
            review_required=True,
            automation_allowed=False,
            review_cadence="on_new_sale",
            decision_reason=(
                "Insufficient series history; wait for a validated category-location "
                "reference before automating an estimate."
            ),
        )

    if policy.demand_segment == DEMAND_SEGMENT_INTERMITTENT:
        return _zero_review_decision(
            "Intermittent demand: review monthly before any replenishment."
        )

    raise ValueError(f"Unknown demand segment: {policy.demand_segment}")


def decide_validated_series_model(
    policy: SeriesPolicy,
    *,
    selected_model_name: str,
    validation_status: str,
    selection_reason: str,
) -> PublicationDecision:
    """Publish the model selected by the per-series backtest.

    All ``validated_*`` routes have three temporal windows. Zero routes with a
    sparse signal remain published forecasts because zero beat the observed
    candidates, but they require monthly review and retain low confidence in the
    per-series policy.
    """
    approved_statuses = {
        "validated_champion",
        "validated_baseline_zero",
        "validated_zero_no_demand",
        "validated_zero_sparse",
    }
    if validation_status in approved_statuses:
        limited_signal = validation_status in {
            "validated_zero_no_demand",
            "validated_zero_sparse",
        }
        return PublicationDecision(
            applied_model_name=selected_model_name,
            applied_strategy=STRATEGY_VALIDATED_SERIES_MODEL,
            publication_status=STATUS_APPROVED_FORECAST,
            review_required=limited_signal,
            automation_allowed=False,
            review_cadence="monthly",
            decision_reason=selection_reason,
        )

    if policy.demand_segment == DEMAND_SEGMENT_NEW:
        return decide_publication(policy, seasonal_baseline_approved=False)
    return _zero_review_decision(selection_reason)


def _zero_review_decision(reason: str) -> PublicationDecision:
    return PublicationDecision(
        applied_model_name=MODEL_BASELINE_ZERO,
        applied_strategy=STRATEGY_ZERO_REVIEW,
        publication_status=STATUS_PLANNING_REVIEW,
        review_required=True,
        automation_allowed=False,
        review_cadence="monthly",
        decision_reason=reason,
    )
