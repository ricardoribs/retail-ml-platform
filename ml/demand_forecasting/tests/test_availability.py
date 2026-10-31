"""Pure tests for retrospective inventory reconstruction."""

from datetime import date, datetime

import pytest

from ml.demand_forecasting.src.availability import (
    StockMovement,
    availability_status,
    reconstruct_monthly_inventory,
)


def test_reconstruction_uses_future_movements_only_to_walk_back_from_snapshot():
    movements = (
        StockMovement(datetime(2026, 9, 5, 10), 10.0),
        StockMovement(datetime(2026, 9, 10, 10), -7.0),
        StockMovement(datetime(2026, 10, 5, 10), 5.0),
        StockMovement(datetime(2026, 10, 20, 10), 100.0),
    )
    result = reconstruct_monthly_inventory(
        snapshot_qty=8.0,
        snapshot_at=datetime(2026, 10, 15),
        movements=movements,
        month=date(2026, 9, 1),
    )
    assert result is not None
    assert result.opening_qty == pytest.approx(0)
    assert result.closing_qty == pytest.approx(3)
    assert result.min_qty == pytest.approx(0)
    assert result.max_qty == pytest.approx(10)
    assert result.movement_count == 2
    assert availability_status(result) == "possible_stockout_within_month"


def test_no_movement_and_zero_balance_means_no_positive_stock():
    result = reconstruct_monthly_inventory(
        snapshot_qty=0,
        snapshot_at=datetime(2026, 10, 1),
        movements=(),
        month=date(2026, 9, 1),
    )
    assert result is not None
    assert availability_status(result) == "no_positive_stock_all_month"


def test_negative_reconstructed_balance_is_invalid_not_a_stockout_claim():
    result = reconstruct_monthly_inventory(
        snapshot_qty=0,
        snapshot_at=datetime(2026, 10, 1),
        movements=(StockMovement(datetime(2026, 9, 10), 2),),
        month=date(2026, 9, 1),
    )
    assert result is not None
    assert result.opening_qty == pytest.approx(-2)
    assert availability_status(result) == "invalid_negative_balance"


def test_snapshot_must_cover_full_month():
    assert (
        reconstruct_monthly_inventory(
            snapshot_qty=1,
            snapshot_at=datetime(2026, 9, 30),
            movements=(),
            month=date(2026, 9, 1),
        )
        is None
    )


def test_simultaneous_movements_are_applied_together():
    timestamp = datetime(2026, 9, 10, 9)
    result = reconstruct_monthly_inventory(
        snapshot_qty=1,
        snapshot_at=datetime(2026, 10, 1),
        movements=(StockMovement(timestamp, -3), StockMovement(timestamp, 3)),
        month=date(2026, 9, 1),
    )
    assert result is not None
    assert result.min_qty == pytest.approx(1)
    assert result.max_qty == pytest.approx(1)
    assert availability_status(result) == "positive_stock_all_month"


def test_rejects_bad_snapshot_and_bad_month():
    with pytest.raises(ValueError, match="primeiro dia"):
        reconstruct_monthly_inventory(
            snapshot_qty=1,
            snapshot_at=datetime(2026, 10, 1),
            movements=(),
            month=date(2026, 9, 30),
        )
    with pytest.raises(ValueError, match="snapshot_qty"):
        reconstruct_monthly_inventory(
            snapshot_qty=-1,
            snapshot_at=datetime(2026, 10, 1),
            movements=(),
            month=date(2026, 9, 1),
        )
