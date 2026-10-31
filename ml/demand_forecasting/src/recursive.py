"""Shared feature construction and recursive forecasting utilities.

Each series state contains only observations before the first forecast
month. After each step, the prediction (never the test actual) is added to
the history used by the next step."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import date
from typing import Any

SeriesKey = tuple[int, int]
SeriesState = dict[str, Any]
PredictionRecord = dict[str, Any]
FROZEN_SIGNAL_COLUMNS: tuple[str, ...] = (
    "last_unit_price",
    "rolling_3m_days_with_sales",
    "rolling_6m_positive_months",
    "months_since_last_positive_sale",
    "received_qty_last_3m",
)


def build_model_input(
    static: Mapping[str, Any],
    history: Sequence[tuple[date, float]],
    target_month: date,
) -> dict[str, Any]:
    """Recreate inference features using only the available history."""
    if not history:
        raise ValueError("Series has no history for forecasting")
    history_months = [month for month, _ in history]
    if history_months != sorted(set(history_months)):
        raise ValueError("History months must be distinct and increasing")
    if history_months[-1] >= target_month:
        raise ValueError("History contains a month equal to or after the target")

    values = [float(value) for _, value in history]
    last_three = values[-3:]
    last_six = values[-6:]
    return {
        "product_variant_id": static["product_variant_id"],
        "category_name": static["category_name"],
        "location_id": static["location_id"],
        "location_type": static["location_type"],
        "state": static["state"],
        "lag_1_qty": values[-1],
        "lag_2_qty": values[-2] if len(values) >= 2 else 0.0,
        "lag_3_qty": values[-3] if len(values) >= 3 else 0.0,
        "rolling_3m_avg_qty": sum(last_three) / len(last_three),
        "rolling_6m_avg_qty": sum(last_six) / len(last_six),
        "month_number": target_month.month,
        "quarter_number": (target_month.month - 1) // 3 + 1,
        "year_number": target_month.year,
        **{column: static.get(column) for column in FROZEN_SIGNAL_COLUMNS},
    }


def recursive_forecast(
    states: Mapping[SeriesKey, SeriesState],
    target_months: Sequence[date],
    predict_batch: Callable[[list[dict[str, Any]]], Mapping[SeriesKey, float]],
) -> list[PredictionRecord]:
    """Forecast months in sequence without mutating the history provided by the caller."""
    if not states:
        raise ValueError("No series available for recursive forecasting")
    if not target_months or list(target_months) != sorted(set(target_months)):
        raise ValueError("Forecast months must be distinct and increasing")

    working_states = {
        key: {"static": dict(state["static"]), "history": list(state["history"])}
        for key, state in states.items()
    }
    for key, state in working_states.items():
        if (
            state["static"]["product_variant_id"],
            state["static"]["location_id"],
        ) != key:
            raise ValueError(f"Static identifiers do not match {key}")
    records: list[PredictionRecord] = []

    for target_month in target_months:
        inputs = [
            {
                **build_model_input(state["static"], state["history"], target_month),
                "_key_product_variant_id": key[0],
                "_key_location_id": key[1],
            }
            for key, state in working_states.items()
        ]
        predictions = predict_batch(inputs)
        if set(predictions) != set(working_states):
            raise ValueError("Forecasts do not exactly cover the expected series")

        for key, state in working_states.items():
            raw_value = float(predictions[key])
            if not math.isfinite(raw_value):
                raise ValueError(f"Forecast is not finite for series {key}")
            forecast_qty = max(raw_value, 0.0)
            state["history"].append((target_month, forecast_qty))
            records.append(
                {
                    "demand_month": target_month,
                    "product_variant_id": key[0],
                    "location_id": key[1],
                    "prediction": forecast_qty,
                }
            )

    return records
