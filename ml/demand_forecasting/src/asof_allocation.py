"""Experimental hierarchical allocation with weights known at the cutoff.

This module does not train or publish models. Every observation used for a
forecast must precede the first target month; realized values are used only
during evaluation."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date

from ml.demand_forecasting.src.hierarchical import (
    ForecastKey,
    GroupKey,
    SeriesKey,
    build_hierarchical_forecasts,
)
from ml.demand_forecasting.src.modeling import shift_months

ADAPTIVE_METHODS = ("seasonal_adaptive", "trend_adaptive")
SERIES_BASELINES = ("zero", "ma3_frozen", "seasonal_series")


def _validate_series_history(
    series_groups: Mapping[SeriesKey, GroupKey],
    series_history: Mapping[SeriesKey, Mapping[date, float]],
    origin: date,
) -> None:
    if set(series_groups) != set(series_history):
        raise ValueError("Per-series history does not exactly cover the cohort")
    for series, history in series_history.items():
        for month, quantity in history.items():
            if month.day != 1 or month >= origin:
                raise ValueError(f"Future history or invalid month in series {series}")
            if not math.isfinite(float(quantity)) or quantity < 0:
                raise ValueError(f"Invalid historical quantity in series {series}")


def history_segment(history: Mapping[date, float], origin: date) -> str:
    """Allocate without reading the holdout; this does not assert inventory availability."""
    for month, quantity in history.items():
        if month.day != 1 or month >= origin or not math.isfinite(float(quantity)):
            raise ValueError("Segmentation history is invalid or in the future")
        if quantity < 0:
            raise ValueError("Historical demand is negative")
    recent = tuple(shift_months(origin, -i) for i in range(1, 7))
    if any(month not in history for month in recent):
        return "historico_incompleto"
    positive = sum(history[month] > 0 for month in recent)
    if positive >= 3:
        return "recorrente_3_6"
    if positive:
        return "ocasional_1_2_6"
    return "sem_venda_6_6"


def build_asof_forecasts(
    *,
    series_groups: Mapping[SeriesKey, GroupKey],
    series_history: Mapping[SeriesKey, Mapping[date, float]],
    group_history: Mapping[GroupKey, Mapping[date, float]],
    target_months: Sequence[date],
) -> dict[str, dict[ForecastKey, float]]:
    """Compare zero, standalone series, and a seasonal group with variable shares.

    The group total follows the audited seasonal/trend rules. Each forecast month,
    half is allocated by each series' share over the latest twelve months and half
    by its share in the same month of the previous year. If one basis has no sales,
    use the other; if both are zero, allocate uniformly. The 50/50 rule is fixed
    and was not optimized on the holdouts.
    """
    targets = tuple(target_months)
    if not targets:
        raise ValueError("Provide at least one target month")
    origin = targets[0]
    _validate_series_history(series_groups, series_history, origin)
    expected_groups: dict[GroupKey, dict[date, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for series, history in series_history.items():
        for month, quantity in history.items():
            expected_groups[series_groups[series]][month] += quantity
    if set(expected_groups) != set(group_history) or any(
        set(monthly) != set(group_history[group])
        or any(
            not math.isclose(quantity, group_history[group][month], abs_tol=1e-6)
            for month, quantity in monthly.items()
        )
        for group, monthly in expected_groups.items()
    ):
        raise ValueError("Group historical totals do not reconcile with the series")
    last12 = tuple(shift_months(origin, -i) for i in range(1, 13))
    recent_qty = {
        series: sum(series_history[series].get(month, 0.0) for month in last12)
        for series in series_groups
    }
    pooled_qty: dict[int, float] = defaultdict(float)
    for (product_id, _location_id), quantity in recent_qty.items():
        pooled_qty[product_id] += quantity
    base = build_hierarchical_forecasts(
        series_groups=series_groups,
        group_history=group_history,
        local_last12=recent_qty,
        pooled_last12=pooled_qty,
        target_months=targets,
    )
    members: dict[GroupKey, list[SeriesKey]] = defaultdict(list)
    for series, group in series_groups.items():
        members[group].append(series)

    predictions: dict[str, dict[ForecastKey, float]] = {
        **base,
        "zero": {},
        "ma3_frozen": {},
        "seasonal_series": {},
        **{method: {} for method in ADAPTIVE_METHODS},
    }
    last3 = tuple(shift_months(origin, -i) for i in range(1, 4))

    for month in targets:
        prior_year = shift_months(month, -12)
        for group, group_members in members.items():
            local_total = sum(recent_qty[series] for series in group_members)
            prior_total = sum(
                series_history[series].get(prior_year, 0.0) for series in group_members
            )
            for series in group_members:
                product_id, location_id = series
                key = (month, product_id, location_id)
                history = series_history[series]
                ma3 = sum(history.get(previous, 0.0) for previous in last3) / 3
                prior = history.get(prior_year)
                predictions["zero"][key] = 0.0
                predictions["ma3_frozen"][key] = ma3
                predictions["seasonal_series"][key] = (
                    prior if prior is not None else ma3
                )

                recent_share = (
                    recent_qty[series] / local_total if local_total > 0 else None
                )
                prior_share = (
                    history.get(prior_year, 0.0) / prior_total
                    if prior_total > 0
                    else None
                )
                if recent_share is not None and prior_share is not None:
                    share = (recent_share + prior_share) / 2
                elif recent_share is not None:
                    share = recent_share
                elif prior_share is not None:
                    share = prior_share
                else:
                    share = 1 / len(group_members)
                for method, source in (
                    ("seasonal_adaptive", "seasonal_uniform"),
                    ("trend_adaptive", "trend_uniform"),
                ):
                    # Uniform weights multiplied by the number of members equal the forecast total.
                    group_total = base[source][key] * len(group_members)
                    predictions[method][key] = group_total * share
    return predictions
