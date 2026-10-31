"""Baselines for the demand forecast.

Evaluates 3-month (`baseline_ma3`) and 6-month (`baseline_ma6`) moving
averages as reference models for regression candidates. The recursive
backtest also compares zero and the same month from the previous year.

The moving averages are precomputed without temporal leakage in
`int_forecast_feature_matrix` (`rolling_3m_avg_qty` and
`rolling_6m_avg_qty`), and are null when a series is too short for a window.
This module applies the fallback cascade: MA6 falls back to MA3 when
unavailable, and both fall back to zero when no average is available. Every
series therefore receives an effective forecast (>= 0) and is evaluated;
`fallback_rows` records how many used a fallback. This module performs no I/O:
`01_baseline.py` reads the view and passes the vectors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ml.demand_forecasting.src.metrics import calculate_metrics
from ml.demand_forecasting.src.recursive import (
    SeriesKey,
    SeriesState,
    recursive_forecast,
)

__all__ = [
    "BASELINE_SOURCE_COLUMNS",
    "FALLBACK_FILL_VALUE",
    "resolve_forecast",
    "resolve_baseline_predictions",
    "evaluate_moving_average_baseline",
    "evaluate_baselines",
    "evaluate_recursive_baselines",
    "recursive_baseline_predictions",
]


BASELINE_SOURCE_COLUMNS: dict[str, str] = {
    "baseline_ma3": "rolling_3m_avg_qty",
    "baseline_ma6": "rolling_6m_avg_qty",
}

# Final cascade fallback when no moving average is available (the series is too
# short even for MA3). Zero is the default for a product-location pair without
# enough history.
FALLBACK_FILL_VALUE = 0.0


def _validate_actual(actual: ArrayLike) -> NDArray[np.float64]:
    """Validate the actual-demand vector: 1D, non-empty, and without NaN/inf."""
    actual_array = np.asarray(actual, dtype=float)
    if actual_array.ndim != 1:
        raise ValueError("actual must be a one-dimensional vector")
    if actual_array.size == 0:
        raise ValueError("actual cannot be empty")
    if not np.all(np.isfinite(actual_array)):
        raise ValueError("actual must contain only finite numeric values")
    return actual_array


def _validate_candidate(
    candidate: ArrayLike,
    expected_size: int,
) -> NDArray[np.float64]:
    """Validate a candidate forecast: 1D, expected size, NaN allowed, inf disallowed."""
    candidate_array = np.asarray(candidate, dtype=float)
    if candidate_array.ndim != 1:
        raise ValueError("each forecast must be a one-dimensional vector")
    if candidate_array.size != expected_size:
        raise ValueError("actual and forecasts must have the same size")
    if np.any(np.isinf(candidate_array)):
        raise ValueError("forecasts cannot contain infinite values")
    return candidate_array


def resolve_forecast(
    primary: ArrayLike,
    *fallbacks: ArrayLike,
    fill_value: float = FALLBACK_FILL_VALUE,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Resolve the effective forecast through a moving-average fallback cascade.

    Use the primary average where available; where it is null (short series), try
    fallback averages in the specified order, then apply `fill_value` (zero). The
    effective forecast never contains NaN and is clipped at zero, consistent with
    the non-negative forecast rule. Also return a mask for rows resolved by a
    fallback (where the primary average was null).
    """
    primary_array = np.asarray(primary, dtype=float)
    resolved = primary_array.copy()
    used_fallback = np.isnan(primary_array)

    for candidate in fallbacks:
        pending = np.isnan(resolved)
        if not pending.any():
            break
        candidate_array = np.asarray(candidate, dtype=float)
        resolved = np.where(pending, candidate_array, resolved)

    resolved = np.where(np.isnan(resolved), fill_value, resolved)
    resolved = np.maximum(resolved, 0.0)
    return resolved, used_fallback


def _baseline_candidate_order(
    ma3_predicted: ArrayLike,
    ma6_predicted: ArrayLike,
) -> dict[str, tuple[ArrayLike, ...]]:
    """Priority order for each baseline (primary, followed by fallbacks).

    - baseline_ma3: MA3 only; final fallback is zero.
    - baseline_ma6: MA6, then MA3 as a fallback; final fallback is zero.
    """
    return {
        "baseline_ma3": (ma3_predicted,),
        "baseline_ma6": (ma6_predicted, ma3_predicted),
    }


def resolve_baseline_predictions(
    ma3_predicted: ArrayLike,
    ma6_predicted: ArrayLike,
) -> dict[str, NDArray[np.float64]]:
    """Return each baseline's effective forecast after the cascade and zero fallback.

    Used by the notebook to write the forecast table with an effective forecast
    (without NaN) on every row, including short series.
    """
    ma3_array = np.asarray(ma3_predicted, dtype=float)
    ma6_array = np.asarray(ma6_predicted, dtype=float)
    if ma3_array.shape != ma6_array.shape:
        raise ValueError("ma3_predicted and ma6_predicted must have the same size")

    order = _baseline_candidate_order(ma3_array, ma6_array)
    return {
        model_name: resolve_forecast(*candidates)[0]
        for model_name, candidates in order.items()
    }


def evaluate_moving_average_baseline(
    actual: ArrayLike,
    model_name: str,
    primary_predicted: ArrayLike,
    *fallback_predicted: ArrayLike,
) -> dict[str, float | int | str]:
    """Evaluate a baseline using the fallback cascade and zero.

    `primary_predicted` is the model's moving average. Where it is null (short
    series), the cascade uses `fallback_predicted` averages in the given order,
    then zero. Every row receives an effective forecast (>= 0) and is evaluated;
    `fallback_rows` counts how many used a fallback.
    """
    actual_array = _validate_actual(actual)
    primary_array = _validate_candidate(primary_predicted, actual_array.size)
    fallback_arrays = tuple(
        _validate_candidate(candidate, actual_array.size)
        for candidate in fallback_predicted
    )

    predicted, used_fallback = resolve_forecast(primary_array, *fallback_arrays)
    fallback_rows = int(np.count_nonzero(used_fallback))

    metrics = calculate_metrics(actual_array, predicted)

    return {"model_name": model_name, **metrics, "fallback_rows": fallback_rows}


def evaluate_baselines(
    actual: ArrayLike,
    ma3_predicted: ArrayLike,
    ma6_predicted: ArrayLike,
) -> dict[str, dict[str, float | int | str]]:
    """Evaluate MA3 and MA6 on the same actual demand, with one entry per model_name."""
    order = _baseline_candidate_order(ma3_predicted, ma6_predicted)
    return {
        model_name: evaluate_moving_average_baseline(actual, model_name, *candidates)
        for model_name, candidates in order.items()
    }


def evaluate_recursive_baselines(
    history_states: Mapping[SeriesKey, SeriesState],
    target_months: Sequence[date],
    actual_rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int | str]]:
    """Evaluate zero, MA3, MA6, and seasonal baselines on the same recursive horizon.

    Series eligible for regression have at least six months of history. The
    seasonal baseline uses the same month from the previous year, or MA3 if missing.
    """
    actual_by_key: dict[tuple[date, int, int], float] = {}
    for row in actual_rows:
        key = (
            row["demand_month"],
            row["product_variant_id"],
            row["location_id"],
        )
        if key in actual_by_key:
            raise ValueError("Duplicate actual rows in the backtest")
        actual_by_key[key] = float(row["demand_qty"])
    if not actual_by_key:
        raise ValueError("The backtest has no actual values")

    seasonal_history = {
        key: {month: qty for month, qty in state["history"]}
        for key, state in history_states.items()
    }
    strategies = (
        "baseline_zero",
        "baseline_ma3",
        "baseline_ma6",
        "baseline_seasonal",
    )
    evaluated: dict[str, dict[str, float | int | str]] = {}
    ordered_keys = sorted(actual_by_key)
    actual = np.asarray([actual_by_key[key] for key in ordered_keys], dtype=float)

    for model_name in strategies:
        fallback_rows = 0

        def predict_batch(
            inputs: list[dict[str, Any]],
        ) -> dict[SeriesKey, float]:
            nonlocal fallback_rows
            predictions = {}
            for row in inputs:
                key = (row["_key_product_variant_id"], row["_key_location_id"])
                if model_name == "baseline_zero":
                    predicted = 0.0
                elif model_name == "baseline_ma3":
                    predicted = row["rolling_3m_avg_qty"]
                elif model_name == "baseline_ma6":
                    predicted = row["rolling_6m_avg_qty"]
                else:
                    prior_year = date(row["year_number"] - 1, row["month_number"], 1)
                    predicted = seasonal_history[key].get(prior_year)
                    if predicted is None:
                        fallback_rows += 1
                        predicted = row["rolling_3m_avg_qty"]
                predictions[key] = float(predicted)
            return predictions

        records = recursive_forecast(history_states, target_months, predict_batch)
        predicted_by_key = {
            (
                row["demand_month"],
                row["product_variant_id"],
                row["location_id"],
            ): row["prediction"]
            for row in records
        }
        if not set(actual_by_key).issubset(predicted_by_key):
            raise ValueError("The recursive baseline did not cover the full test window")
        predicted = np.asarray(
            [predicted_by_key[key] for key in ordered_keys], dtype=float
        )
        evaluated[model_name] = {
            "model_name": model_name,
            **calculate_metrics(actual, predicted),
            "fallback_rows": fallback_rows,
        }

    return evaluated


def recursive_baseline_predictions(
    history_states: Mapping[SeriesKey, SeriesState],
    target_months: Sequence[date],
) -> dict[str, list[dict[str, Any]]]:
    """Generate recursive forecasts for each baseline without using actuals.

    This routine is shared by granular backtesting and inference. The seasonal
    fallback to MA3 matches the one used in the global evaluation.
    """
    seasonal_history = {
        key: {month: qty for month, qty in state["history"]}
        for key, state in history_states.items()
    }
    strategies = (
        "baseline_zero",
        "baseline_ma3",
        "baseline_ma6",
        "baseline_seasonal",
    )
    predicted_by_model: dict[str, list[dict[str, Any]]] = {}

    for model_name in strategies:

        def predict_batch(
            inputs: list[dict[str, Any]],
        ) -> dict[SeriesKey, float]:
            predictions = {}
            for row in inputs:
                key = (row["_key_product_variant_id"], row["_key_location_id"])
                if model_name == "baseline_zero":
                    predicted = 0.0
                elif model_name == "baseline_ma3":
                    predicted = row["rolling_3m_avg_qty"]
                elif model_name == "baseline_ma6":
                    predicted = row["rolling_6m_avg_qty"]
                else:
                    prior_year = date(row["year_number"] - 1, row["month_number"], 1)
                    predicted = seasonal_history[key].get(
                        prior_year, row["rolling_3m_avg_qty"]
                    )
                predictions[key] = float(predicted)
            return predictions

        predicted_by_model[model_name] = recursive_forecast(
            history_states, target_months, predict_batch
        )

    return predicted_by_model
