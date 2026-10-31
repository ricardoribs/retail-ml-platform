"""Testes do contrato temporal da feature matrix de forecast."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

FEATURE_MATRIX_PATH = (
    Path(__file__).resolve().parents[3]
    / "dbt"
    / "models"
    / "intermediate"
    / "int_forecast_feature_matrix.sql"
)


@pytest.fixture(scope="module")
def normalized_feature_matrix_sql() -> str:
    """Read and normalize the actual SQL to validate its temporal contract."""
    sql = FEATURE_MATRIX_PATH.read_text(encoding="utf-8").lower()
    return re.sub(r"\s+", " ", sql)


@pytest.mark.parametrize("lag_months", [1, 2, 3])
def test_lags_use_only_previous_months_from_the_same_series(
    normalized_feature_matrix_sql: str,
    lag_months: int,
) -> None:
    """Ensure each lag reads a prior month from the same series."""
    expected_lag = (
        f"lag(demand_qty, {lag_months}) over ( "
        "partition by product_variant_id, location_id "
        "order by demand_month "
        f") as lag_{lag_months}_qty"
    )

    assert expected_lag in normalized_feature_matrix_sql


def test_rolling_windows_exclude_the_target_month(
    normalized_feature_matrix_sql: str,
) -> None:
    """Ensure target-month demand is excluded from historical features."""
    three_month_window = "rows between 3 preceding and 1 preceding"
    six_month_window = "rows between 6 preceding and 1 preceding"

    assert three_month_window in normalized_feature_matrix_sql
    assert normalized_feature_matrix_sql.count(three_month_window) == 2
    assert normalized_feature_matrix_sql.count(six_month_window) == 1
    assert "current row" not in normalized_feature_matrix_sql
