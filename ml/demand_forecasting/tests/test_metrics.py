"""Unit tests for demand forecast metrics."""

from __future__ import annotations

import math

import pytest

from ml.demand_forecasting.src.metrics import calculate_metrics, mae, mape, rmse, wape


def test_metrics_return_expected_values():
    actual = [100, 200]
    predicted = [110, 180]

    assert mae(actual, predicted) == pytest.approx(15.0)
    assert rmse(actual, predicted) == pytest.approx(math.sqrt(250))
    assert mape(actual, predicted) == pytest.approx(10.0)
    assert wape(actual, predicted) == pytest.approx(10.0)


def test_mape_ignores_rows_with_zero_actual_demand():
    actual = [0, 100]
    predicted = [20, 90]

    assert mape(actual, predicted) == pytest.approx(10.0)


def test_wape_keeps_zero_demand_errors_in_numerator():
    actual = [0, 100]
    predicted = [20, 90]

    assert wape(actual, predicted) == pytest.approx(30.0)


def test_percentage_metrics_are_nan_when_all_actual_values_are_zero():
    actual = [0, 0]
    predicted = [0, 5]

    assert math.isnan(mape(actual, predicted))
    assert math.isnan(wape(actual, predicted))


def test_calculate_metrics_returns_persistence_contract():
    result = calculate_metrics([100, 200], [110, 180])

    assert result == pytest.approx(
        {
            "mae": 15.0,
            "rmse": math.sqrt(250),
            "mape": 10.0,
            "wape": 10.0,
            "evaluated_rows": 2,
        }
    )


@pytest.mark.parametrize(
    ("actual", "predicted", "message"),
    [
        ([], [], "cannot be empty"),
        ([1, 2], [1], "same size"),
        ([[1, 2]], [[1, 2]], "one-dimensional vectors"),
        ([1, float("nan")], [1, 2], "actual"),
        ([1, 2], [1, float("inf")], "predicted"),
    ],
)
def test_invalid_inputs_raise_value_error(actual, predicted, message):
    with pytest.raises(ValueError, match=message):
        calculate_metrics(actual, predicted)
