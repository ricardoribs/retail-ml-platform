"""Diagnostic reconstruction of monthly inventory from balances and movements.

For retrospective audits only. A current balance and a movement ledger do
not, by themselves, prove that the history is complete or that reconstructed
positions were available in real time at historical cutoffs."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time

from ml.demand_forecasting.src.modeling import shift_months


@dataclass(frozen=True)
class StockMovement:
    occurred_at: datetime
    quantity: float


@dataclass(frozen=True)
class MonthlyInventory:
    opening_qty: float
    closing_qty: float
    min_qty: float
    max_qty: float
    movement_count: int


def reconstruct_monthly_inventory(
    *,
    snapshot_qty: float,
    snapshot_at: datetime,
    movements: Sequence[StockMovement],
    month: date,
) -> MonthlyInventory | None:
    """Reconstruct the full month from the current balance using signed movements.

    Return ``None`` when the snapshot predates the end of the month. The monthly
    maximum includes the opening balance and the balance after each movement
    timestamp; simultaneous movements are applied together.
    """
    if month.day != 1:
        raise ValueError("month must be the first day of the month")
    if snapshot_at.tzinfo is not None:
        raise ValueError("snapshot_at must use timezone-naive timestamps")
    snapshot = float(snapshot_qty)
    if not math.isfinite(snapshot) or snapshot < 0:
        raise ValueError("snapshot_qty must be finite and non-negative")

    month_start = datetime.combine(month, time.min)
    next_month = datetime.combine(shift_months(month, 1), time.min)
    if snapshot_at < next_month:
        return None

    monthly_events: dict[datetime, float] = defaultdict(float)
    movement_count = 0
    future_change = 0.0
    for movement in movements:
        if movement.occurred_at.tzinfo is not None:
            raise ValueError("Movements must use timezone-naive timestamps")
        quantity = float(movement.quantity)
        if not math.isfinite(quantity):
            raise ValueError("Movement quantity is not finite")
        if movement.occurred_at < month_start or movement.occurred_at > snapshot_at:
            continue
        if movement.occurred_at >= next_month:
            future_change += quantity
        else:
            monthly_events[movement.occurred_at] += quantity
            movement_count += 1

    closing = snapshot - future_change
    opening = closing - sum(monthly_events.values())
    balance = opening
    minimum = opening
    maximum = opening
    for occurred_at in sorted(monthly_events):
        balance += monthly_events[occurred_at]
        minimum = min(minimum, balance)
        maximum = max(maximum, balance)
    if not math.isclose(balance, closing, abs_tol=1e-6):
        raise AssertionError("Monthly movements did not reconcile")
    return MonthlyInventory(
        opening_qty=opening,
        closing_qty=closing,
        min_qty=minimum,
        max_qty=maximum,
        movement_count=movement_count,
    )


def availability_status(
    inventory: MonthlyInventory | None,
    *,
    tolerance: float = 1e-6,
) -> str:
    """Classify likely availability without inferring unobserved demand."""
    if inventory is None:
        return "snapshot_before_month_end"
    if inventory.min_qty < -tolerance:
        return "invalid_negative_balance"
    if inventory.max_qty <= tolerance:
        return "no_positive_stock_all_month"
    if inventory.min_qty <= tolerance:
        return "possible_stockout_within_month"
    return "positive_stock_all_month"
