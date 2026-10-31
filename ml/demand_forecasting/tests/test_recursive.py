"""Pure tests for recursion and baselines, without a Spark dependency."""

from __future__ import annotations

from datetime import date

import pytest

from ml.demand_forecasting.src.baseline import evaluate_recursive_baselines
from ml.demand_forecasting.src.recursive import (
    build_model_input,
    recursive_forecast,
)


def _states():
    return {
        (1, 10): {
            "static": {
                "product_variant_id": 1,
                "category_name": "motor",
                "location_id": 10,
                "location_type": "store",
                "state": "SP",
                "last_unit_price": 120.0,
                "rolling_3m_days_with_sales": 12.0,
                "rolling_6m_positive_months": 4.0,
                "months_since_last_positive_sale": 0.0,
                "received_qty_last_3m": 8.0,
            },
            "history": [(date(2026, month, 1), float(month)) for month in range(1, 7)],
        }
    }


def test_build_model_input_uses_only_previous_months():
    state = _states()[(1, 10)]
    features = build_model_input(state["static"], state["history"], date(2026, 7, 1))

    assert features["lag_1_qty"] == 6.0
    assert features["lag_2_qty"] == 5.0
    assert features["lag_3_qty"] == 4.0
    assert features["rolling_3m_avg_qty"] == 5.0
    assert features["rolling_6m_avg_qty"] == 3.5
    assert features["quarter_number"] == 3
    assert features["last_unit_price"] == 120.0
    assert features["received_qty_last_3m"] == 8.0


def test_recursive_forecast_feeds_predictions_and_keeps_original_state():
    states = _states()
    seen_lags = []

    def predict_batch(inputs):
        seen_lags.append(inputs[0]["lag_1_qty"])
        return {(1, 10): inputs[0]["lag_1_qty"] + 1}

    records = recursive_forecast(
        states, (date(2026, 7, 1), date(2026, 8, 1)), predict_batch
    )

    assert seen_lags == [6.0, 7.0]
    assert [row["prediction"] for row in records] == [7.0, 8.0]
    assert len(states[(1, 10)]["history"]) == 6


def test_recursive_forecast_keeps_as_of_signals_frozen_across_horizons():
    seen_prices = []

    def predict_batch(inputs):
        seen_prices.append(inputs[0]["last_unit_price"])
        return {(1, 10): 1.0}

    recursive_forecast(_states(), (date(2026, 7, 1), date(2026, 8, 1)), predict_batch)

    assert seen_prices == [120.0, 120.0]


def test_recursive_forecast_rejects_leakage_and_incomplete_results():
    states = _states()
    states[(1, 10)]["history"].append((date(2026, 7, 1), 999.0))
    with pytest.raises(ValueError, match="posterior ao alvo"):
        recursive_forecast(states, (date(2026, 7, 1),), lambda inputs: {})

    with pytest.raises(ValueError, match="expected series"):
        recursive_forecast(_states(), (date(2026, 7, 1),), lambda inputs: {})


def test_recursive_baselines_include_zero_and_do_not_use_test_actual_as_lag():
    actual_rows = [
        {
            "demand_month": date(2026, 7, 1),
            "product_variant_id": 1,
            "location_id": 10,
            "demand_qty": 10.0,
        },
        {
            "demand_month": date(2026, 8, 1),
            "product_variant_id": 1,
            "location_id": 10,
            "demand_qty": 20.0,
        },
    ]
    results = evaluate_recursive_baselines(
        _states(), (date(2026, 7, 1), date(2026, 8, 1)), actual_rows
    )

    assert results["baseline_zero"]["wape"] == 100.0
    assert results["baseline_ma3"]["mae"] == pytest.approx(
        (abs(10 - 5) + abs(20 - (5 + 6 + 5) / 3)) / 2
    )
    assert results["baseline_ma6"]["mae"] == pytest.approx(
        (abs(10 - 3.5) + abs(20 - (2 + 3 + 4 + 5 + 6 + 3.5) / 6)) / 2
    )
    assert results["baseline_seasonal"]["fallback_rows"] == 2
    assert results["baseline_seasonal"]["mae"] == results["baseline_ma3"]["mae"]


def test_seasonal_baseline_uses_prior_year_without_test_actual():
    states = _states()
    states[(1, 10)]["history"] = [
        (date(2025, month, 1), float(month * 10)) for month in range(1, 13)
    ]
    actual_rows = [
        {
            "demand_month": date(2026, month, 1),
            "product_variant_id": 1,
            "location_id": 10,
            "demand_qty": float(month * 10),
        }
        for month in (1, 2)
    ]

    results = evaluate_recursive_baselines(
        states, (date(2026, 1, 1), date(2026, 2, 1)), actual_rows
    )

    assert results["baseline_seasonal"]["wape"] == 0.0
    assert results["baseline_seasonal"]["fallback_rows"] == 0
