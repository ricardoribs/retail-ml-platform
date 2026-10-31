"""Deterministic forecasters for intermittent monthly demand.

Croston, SBA, and TSB update only from sales observed before the forecast
origin. They are backtest candidates: they do not select hyperparameters from
the holdout or publish forecasts on their own."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import date

SeriesKey = tuple[int, int]
ForecastKey = tuple[date, int, int]

INTERMITTENT_METHODS = (
    "croston_classic_a010",
    "croston_sba_a010",
    "tsb_a010_b010",
)
CROSTON_ALPHA = 0.10
TSB_BETA = 0.10


def _next_month(month: date) -> date:
    """Return the first day of the following month."""
    return date(month.year + (month.month == 12), month.month % 12 + 1, 1)


def _validate_targets(target_months: Sequence[date]) -> tuple[date, ...]:
    targets = tuple(target_months)
    if not targets or any(month.day != 1 for month in targets):
        raise ValueError("Target months must be first days of months")
    if tuple(sorted(set(targets))) != targets:
        raise ValueError("Target months must be distinct and ordered")
    if any(
        current != _next_month(previous)
        for previous, current in zip(targets, targets[1:])
    ):
        raise ValueError("Target months must be consecutive")
    return targets


def _monthly_values(history: Mapping[date, float], origin: date) -> list[float]:
    """Fill monthly gaps with zero and reject future observations."""
    if not history:
        return []
    if any(month.day != 1 or month >= origin for month in history):
        raise ValueError("History contains a future or invalid month")
    if any(not math.isfinite(float(value)) or value < 0 for value in history.values()):
        raise ValueError("History contains an invalid quantity")

    month = min(history)
    values = []
    while month < origin:
        values.append(float(history.get(month, 0.0)))
        month = _next_month(month)
    return values


def croston_forecast(values: Sequence[float], alpha: float = CROSTON_ALPHA) -> float:
    """Estimate average demand per period using the classic Croston method."""
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be in the interval (0, 1]")
    demand_size: float | None = None
    demand_interval: float | None = None
    periods_since_sale = 0

    for value in values:
        quantity = float(value)
        if not math.isfinite(quantity) or quantity < 0:
            raise ValueError("Values must be finite and non-negative")
        periods_since_sale += 1
        if quantity <= 0:
            continue
        if demand_size is None or demand_interval is None:
            demand_size = quantity
            demand_interval = float(periods_since_sale)
        else:
            demand_size += alpha * (quantity - demand_size)
            demand_interval += alpha * (periods_since_sale - demand_interval)
        periods_since_sale = 0

    if demand_size is None or demand_interval is None:
        return 0.0
    return max(demand_size / demand_interval, 0.0)


def sba_forecast(values: Sequence[float], alpha: float = CROSTON_ALPHA) -> float:
    """Apply the SBA bias correction to the classic Croston forecast."""
    return (1 - alpha / 2) * croston_forecast(values, alpha)


def tsb_forecast(
    values: Sequence[float],
    alpha: float = CROSTON_ALPHA,
    beta: float = TSB_BETA,
) -> float:
    """Estimate expected demand using the TSB method."""
    probability, demand_size = tsb_components(values, alpha=alpha, beta=beta)
    return max(probability * demand_size, 0.0)


def tsb_components(
    values: Sequence[float],
    alpha: float = CROSTON_ALPHA,
    beta: float = TSB_BETA,
) -> tuple[float, float]:
    """Return TSB's ``P(sale)`` and ``E[quantity | sale]`` components.

    Separating the components makes it possible to evaluate sales discrimination
    separately from predicted magnitude. The function preserves the exact update
    used by :func:`tsb_forecast`; it does not calibrate or apply a decision threshold.
    """
    if not 0 < alpha <= 1 or not 0 < beta <= 1:
        raise ValueError("alpha and beta must be in the interval (0, 1]")
    probability: float | None = None
    demand_size: float | None = None

    for value in values:
        quantity = float(value)
        if not math.isfinite(quantity) or quantity < 0:
            raise ValueError("Values must be finite and non-negative")
        occurrence = float(quantity > 0)
        probability = (
            occurrence
            if probability is None
            else probability + beta * (occurrence - probability)
        )
        if quantity > 0:
            demand_size = (
                quantity
                if demand_size is None
                else demand_size + alpha * (quantity - demand_size)
            )

    if probability is None or demand_size is None:
        return 0.0, 0.0
    return max(probability, 0.0), max(demand_size, 0.0)


def tsb_occurrence_probability(
    values: Sequence[float],
    alpha: float = CROSTON_ALPHA,
    beta: float = TSB_BETA,
) -> float:
    """Return only the probability of occurrence estimated by TSB."""
    probability, _ = tsb_components(values, alpha=alpha, beta=beta)
    return probability


def build_intermittent_forecasts(
    *,
    series_history: Mapping[SeriesKey, Mapping[date, float]],
    target_months: Sequence[date],
) -> dict[str, dict[ForecastKey, float]]:
    """Generate three fixed candidates without test-horizon actuals.

    Since there are no observations between future months, each candidate's
    expectation remains constant across the horizon. This differs from feeding
    the method zero or its own forecast, which would create a nonexistent
    observation and distort the backtest.
    """
    targets = _validate_targets(target_months)
    origin = targets[0]
    forecasts: dict[str, dict[ForecastKey, float]] = {
        method: {} for method in INTERMITTENT_METHODS
    }
    for (product_id, location_id), history in series_history.items():
        values = _monthly_values(history, origin)
        predicted = {
            "croston_classic_a010": croston_forecast(values),
            "croston_sba_a010": sba_forecast(values),
            "tsb_a010_b010": tsb_forecast(values),
        }
        for target in targets:
            key = (target, product_id, location_id)
            for method, value in predicted.items():
                forecasts[method][key] = value
    return forecasts


def build_tsb_occurrence_probabilities(
    *,
    series_history: Mapping[SeriesKey, Mapping[date, float]],
    target_months: Sequence[date],
) -> dict[ForecastKey, float]:
    """Generate TSB probabilities frozen at the horizon origin.

    Without a new observation between future months, the probability remains the
    same across the horizon. This is intentional and makes explicit that this
    candidate has no calendar signal or monthly covariates.
    """
    targets = _validate_targets(target_months)
    origin = targets[0]
    probabilities: dict[ForecastKey, float] = {}
    for (product_id, location_id), history in series_history.items():
        probability = tsb_occurrence_probability(_monthly_values(history, origin))
        for target in targets:
            probabilities[(target, product_id, location_id)] = probability
    return probabilities
