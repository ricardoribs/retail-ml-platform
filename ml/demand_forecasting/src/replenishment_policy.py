"""Manual replenishment policy, separate from the numeric forecast.

Recommendations use only the inventory snapshot: balance, lead time, reorder
point, coverage, and open purchase orders. They never interpret a zero
forecast fallback as future demand."""

from __future__ import annotations

from dataclasses import dataclass

from ml.demand_forecasting.src.segmentation import (
    DEMAND_SEGMENT_INTERMITTENT,
    DEMAND_SEGMENT_NEW,
    DEMAND_SEGMENT_SEASONAL,
    DEMAND_SEGMENT_STABLE,
)

PRIORITY_CRITICAL = "critical"
PRIORITY_HIGH = "high"
PRIORITY_MEDIUM = "medium"
PRIORITY_LOW = "low"


@dataclass(frozen=True)
class ReplenishmentDecision:
    """Planning action that always requires human validation."""

    replenishment_strategy: str
    priority: str
    action_required: bool
    review_cadence: str
    decision_reason: str


def decide_replenishment(
    *,
    demand_segment: str,
    is_reorder_needed: bool,
    is_reorder_needed_after_inbound: bool,
    is_stockout_risk: bool,
    suggested_purchase_qty: float | None,
) -> ReplenishmentDecision:
    """Prioritize replenishment based on inventory rather than the technical forecast."""
    strategy = _strategy_for_segment(demand_segment)
    suggested_qty = float(suggested_purchase_qty or 0.0)

    if is_stockout_risk:
        return ReplenishmentDecision(
            replenishment_strategy=strategy,
            priority=PRIORITY_CRITICAL,
            action_required=True,
            review_cadence="daily",
            decision_reason=(
                "Inventory coverage is shorter than lead time; verify physical stock, "
                "open purchases, and replenishment urgency."
            ),
        )

    if is_reorder_needed_after_inbound:
        return ReplenishmentDecision(
            replenishment_strategy=strategy,
            priority=PRIORITY_HIGH,
            action_required=True,
            review_cadence="weekly",
            decision_reason=(
                "Projected inventory, including open purchases, is at or below the "
                "reorder point."
            ),
        )

    if is_reorder_needed:
        return ReplenishmentDecision(
            replenishment_strategy=strategy,
            priority=PRIORITY_HIGH,
            action_required=True,
            review_cadence="weekly",
            decision_reason=(
                "Current inventory is at or below the reorder point; check whether "
                "open purchases cover the need."
            ),
        )

    if suggested_qty > 0:
        return ReplenishmentDecision(
            replenishment_strategy=strategy,
            priority=PRIORITY_MEDIUM,
            action_required=True,
            review_cadence="weekly",
            decision_reason=(
                "The inventory coverage policy suggests a purchase; validate before "
                "placing an order."
            ),
        )

    return ReplenishmentDecision(
        replenishment_strategy=strategy,
        priority=PRIORITY_LOW,
        action_required=False,
        review_cadence=(
            "monthly" if demand_segment != DEMAND_SEGMENT_NEW else "on_new_sale"
        ),
        decision_reason=(
            "No replenishment alert in the snapshot; continue monitoring inventory "
            "and coverage."
        ),
    )


def _strategy_for_segment(demand_segment: str) -> str:
    strategies = {
        DEMAND_SEGMENT_STABLE: "reorder_point_manual_review",
        DEMAND_SEGMENT_SEASONAL: "seasonal_reorder_point_manual_review",
        DEMAND_SEGMENT_INTERMITTENT: "intermittent_manual_reorder_review",
        DEMAND_SEGMENT_NEW: "new_item_inventory_review",
    }
    try:
        return strategies[demand_segment]
    except KeyError as exc:
        raise ValueError(f"Unknown demand segment: {demand_segment}") from exc
