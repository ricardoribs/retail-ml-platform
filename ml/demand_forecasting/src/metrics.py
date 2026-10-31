"""Evaluation metrics for the demand forecast.

The functions in this module accept one-dimensional vectors of actual and
predicted values. MAPE and WAPE are returned as percentages (0 to 100).

Rules when actual demand is zero:

- MAPE ignores these observations because individual percentage error is
  undefined. If no actual demand value is nonzero, the result is `numpy.nan`.
- WAPE retains all absolute errors in the numerator. If the sum of absolute
  actual demand is zero, the result is `numpy.nan`."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

MetricInput = ArrayLike | Sequence[float]


def _validate_inputs(
    actual: MetricInput,
    predicted: MetricInput,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert and validate the vectors used to calculate metrics."""
    actual_array = np.asarray(actual, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)

    if actual_array.ndim != 1 or predicted_array.ndim != 1:
        raise ValueError("actual and predicted must be one-dimensional vectors")
    if actual_array.size == 0:
        raise ValueError("actual and predicted cannot be empty")
    if actual_array.shape != predicted_array.shape:
        raise ValueError("actual and predicted must have the same size")
    if not np.all(np.isfinite(actual_array)):
        raise ValueError("actual must contain only finite numeric values")
    if not np.all(np.isfinite(predicted_array)):
        raise ValueError("predicted must contain only finite numeric values")

    return actual_array, predicted_array


def mae(actual: MetricInput, predicted: MetricInput) -> float:
    """Calculate mean absolute error in the same units as demand."""
    actual_array, predicted_array = _validate_inputs(actual, predicted)
    return float(np.mean(np.abs(actual_array - predicted_array)))


def rmse(actual: MetricInput, predicted: MetricInput) -> float:
    """Calculate root mean squared error."""
    actual_array, predicted_array = _validate_inputs(actual, predicted)
    squared_errors = np.square(actual_array - predicted_array)
    return float(np.sqrt(np.mean(squared_errors)))


def mape(actual: MetricInput, predicted: MetricInput) -> float:
    """Calculate mean absolute percentage error, ignoring zero actuals.

    Return ``numpy.nan`` when all actual observations are zero.
    """
    actual_array, predicted_array = _validate_inputs(actual, predicted)
    nonzero_mask = actual_array != 0

    if not np.any(nonzero_mask):
        return float("nan")

    percentage_errors = np.abs(
        (actual_array[nonzero_mask] - predicted_array[nonzero_mask])
        / actual_array[nonzero_mask]
    )
    return float(np.mean(percentage_errors) * 100)


def wape(actual: MetricInput, predicted: MetricInput) -> float:
    """Calculate weighted absolute percentage error.

    Return ``numpy.nan`` when the sum of absolute actual values is zero.
    """
    actual_array, predicted_array = _validate_inputs(actual, predicted)
    denominator = np.sum(np.abs(actual_array))

    if denominator == 0:
        return float("nan")

    absolute_error = np.sum(np.abs(actual_array - predicted_array))
    return float((absolute_error / denominator) * 100)


def calculate_metrics(
    actual: MetricInput,
    predicted: MetricInput,
) -> dict[str, float | int]:
    """Calculate all forecast metrics in a single call."""
    actual_array, predicted_array = _validate_inputs(actual, predicted)

    return {
        "mae": mae(actual_array, predicted_array),
        "rmse": rmse(actual_array, predicted_array),
        "mape": mape(actual_array, predicted_array),
        "wape": wape(actual_array, predicted_array),
        "evaluated_rows": int(actual_array.size),
    }
