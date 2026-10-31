"""Tests for the manual replenishment policy."""

from ml.demand_forecasting.src.replenishment_policy import (
    PRIORITY_CRITICAL,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    decide_replenishment,
)


def test_stockout_risk_has_priority_over_every_other_inventory_signal():
    decision = decide_replenishment(
        demand_segment="intermittent",
        is_reorder_needed=False,
        is_reorder_needed_after_inbound=False,
        is_stockout_risk=True,
        suggested_purchase_qty=0,
    )

    assert decision.priority == PRIORITY_CRITICAL
    assert decision.action_required is True
    assert decision.review_cadence == "daily"
    assert decision.replenishment_strategy == "intermittent_manual_reorder_review"


def test_projected_reorder_alert_requires_weekly_manual_review():
    decision = decide_replenishment(
        demand_segment="seasonal",
        is_reorder_needed=True,
        is_reorder_needed_after_inbound=True,
        is_stockout_risk=False,
        suggested_purchase_qty=18,
    )

    assert decision.priority == PRIORITY_HIGH
    assert decision.action_required is True
    assert decision.review_cadence == "weekly"
    assert decision.replenishment_strategy == "seasonal_reorder_point_manual_review"


def test_no_inventory_alert_is_monitored_without_using_forecast_zero():
    decision = decide_replenishment(
        demand_segment="stable",
        is_reorder_needed=False,
        is_reorder_needed_after_inbound=False,
        is_stockout_risk=False,
        suggested_purchase_qty=None,
    )

    assert decision.priority == PRIORITY_LOW
    assert decision.action_required is False
    assert decision.replenishment_strategy == "reorder_point_manual_review"
