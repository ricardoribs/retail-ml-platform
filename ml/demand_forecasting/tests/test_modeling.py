"""Pure tests for the modeling contract, without a Spark dependency."""

from __future__ import annotations

from datetime import date

import pytest

from ml.demand_forecasting.src.modeling import (
    CATEGORICAL_FEATURE_COLUMNS,
    INTERMITTENT_MODEL_SPECS,
    NUMERIC_FEATURE_COLUMNS,
    REGRESSION_MODEL_SPECS,
    build_temporal_split,
    select_best_model,
    select_deployable_regression,
)


def test_temporal_split_reserves_last_three_closed_months():
    split = build_temporal_split(date(2026, 8, 31), test_months=3)

    assert split.train_end == date(2026, 5, 31)
    assert split.test_start == date(2026, 6, 1)
    assert split.test_end == date(2026, 8, 31)


def test_temporal_split_rejects_nonpositive_window():
    with pytest.raises(ValueError, match="maior que zero"):
        build_temporal_split(date(2026, 8, 31), test_months=0)


def test_model_specs_cover_required_regression_families():
    assert {spec.family for spec in REGRESSION_MODEL_SPECS} == {
        "ridge",
        "random_forest",
        "gradient_boosted_trees",
    }


def test_l1_candidate_is_experimental_and_optimizes_absolute_error():
    [candidate] = INTERMITTENT_MODEL_SPECS

    assert candidate.model_name == "gradient_boosted_trees_l1"
    assert candidate.family == "gradient_boosted_trees"
    assert candidate.parameters["lossType"] == "absolute"
    assert candidate.model_name not in {
        spec.model_name for spec in REGRESSION_MODEL_SPECS
    }


def test_features_cover_lags_rollings_calendar_product_and_location():
    assert {
        "lag_1_qty",
        "lag_2_qty",
        "lag_3_qty",
        "rolling_3m_avg_qty",
        "rolling_6m_avg_qty",
        "month_number",
        "quarter_number",
        "year_number",
    }.issubset(NUMERIC_FEATURE_COLUMNS)
    assert {
        "last_unit_price",
        "rolling_3m_days_with_sales",
        "rolling_6m_positive_months",
        "months_since_last_positive_sale",
        "received_qty_last_3m",
    }.issubset(NUMERIC_FEATURE_COLUMNS)
    assert {
        "product_variant_id",
        "category_name",
        "location_id",
        "state",
    }.issubset(CATEGORICAL_FEATURE_COLUMNS)


def test_best_model_uses_wape_then_mae_then_rmse():
    rows = [
        {"model_name": "a", "wape": 20.0, "mae": 2.0, "rmse": 3.0},
        {"model_name": "b", "wape": 10.0, "mae": 4.0, "rmse": 5.0},
        {"model_name": "c", "wape": 10.0, "mae": 3.0, "rmse": 6.0},
    ]

    assert select_best_model(rows)["model_name"] == "c"


def test_best_model_can_select_zero_demand_baseline_globally():
    rows = [
        {"model_name": "baseline_zero", "wape": 100.0, "mae": 0.75, "rmse": 2.42},
        {
            "model_name": "baseline_seasonal",
            "wape": 176.73,
            "mae": 1.33,
            "rmse": 3.19,
        },
        {
            "model_name": "gradient_boosted_trees",
            "wape": 176.79,
            "mae": 1.33,
            "rmse": 2.30,
        },
    ]

    assert select_best_model(rows)["model_name"] == "baseline_zero"


def test_best_model_rejects_nonfinite_metrics():
    with pytest.raises(ValueError, match="finite metrics"):
        select_best_model(
            [{"model_name": "invalid", "wape": float("nan"), "mae": 1, "rmse": 1}]
        )


def _quality_rows(regression_wape: float) -> list[dict]:
    return [
        {
            "model_name": spec.model_name,
            "wape": regression_wape if index == 0 else regression_wape + 5,
            "mae": 1.0,
            "rmse": 2.0,
        }
        for index, spec in enumerate(REGRESSION_MODEL_SPECS)
    ] + [
        {"model_name": "baseline_zero", "wape": 100.0, "mae": 1, "rmse": 2},
        {"model_name": "baseline_ma3", "wape": 120.0, "mae": 1, "rmse": 2},
        {"model_name": "baseline_ma6", "wape": 130.0, "mae": 1, "rmse": 2},
        {"model_name": "baseline_seasonal", "wape": 110.0, "mae": 1, "rmse": 2},
    ]


def test_deployable_regression_must_beat_zero_and_moving_averages():
    best, reference = select_deployable_regression(_quality_rows(90.0))

    assert best["wape"] == 90.0
    assert reference["model_name"] == "baseline_zero"


def test_deployable_regression_rejects_current_error_level():
    with pytest.raises(ValueError, match="Forecast not published"):
        select_deployable_regression(_quality_rows(176.79))


def test_experimental_candidate_cannot_bypass_operational_quality_gate():
    [candidate] = INTERMITTENT_MODEL_SPECS
    rows = _quality_rows(176.79)
    rows.append(
        {"model_name": candidate.model_name, "wape": 20.0, "mae": 0.2, "rmse": 0.5}
    )

    with pytest.raises(ValueError, match="Forecast not published"):
        select_deployable_regression(rows)


def test_deployable_regression_must_beat_seasonal_reference_too():
    rows = _quality_rows(90.0)
    rows[-1]["wape"] = 80.0
    with pytest.raises(ValueError, match="baseline_seasonal"):
        select_deployable_regression(rows)


def test_deployable_regression_requires_all_comparators():
    with pytest.raises(ValueError, match="incompleta"):
        select_deployable_regression(_quality_rows(90.0)[:-1])


def test_deployable_regression_rejects_duplicate_model_rows():
    rows = _quality_rows(90.0)
    with pytest.raises(ValueError, match="duplicados"):
        select_deployable_regression([*rows, rows[0]])
