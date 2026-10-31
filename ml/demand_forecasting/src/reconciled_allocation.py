"""SKU-location forecast reconciled across product and location margins.

The SKU-location grain is sparse, so independent forecasts are unreliable
for many cells. This candidate first forecasts the denser margins (product
across the network and each location total), then reconciles a historical
share matrix to satisfy both. All information precedes the first target
month; the module performs no I/O and does not publish forecasts."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date

from ml.demand_forecasting.src.hierarchical import ForecastKey, GroupKey, SeriesKey
from ml.demand_forecasting.src.modeling import shift_months

RECONCILED_ALLOCATION_METHOD = "product_location_seasonal_ipf"
_EPSILON = 1e-12


def _validate_target_months(target_months: Sequence[date]) -> tuple[date, ...]:
    targets = tuple(target_months)
    if not targets or any(month.day != 1 for month in targets):
        raise ValueError("target_months must contain first days of months")
    expected = tuple(shift_months(targets[0], offset) for offset in range(len(targets)))
    if targets != expected:
        raise ValueError("target_months must be consecutive and ordered")
    return targets


def _finite_nonnegative(value: float, field: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return numeric


def _validate_history(
    series_groups: Mapping[SeriesKey, GroupKey],
    series_history: Mapping[SeriesKey, Mapping[date, float]],
    origin: date,
) -> None:
    if not series_groups or set(series_groups) != set(series_history):
        raise ValueError("History and groups must exactly cover the series")
    for (product_id, location_id), group in series_groups.items():
        if group[0] != location_id:
            raise ValueError("The series location differs from the group location")
        if not isinstance(product_id, int):
            raise ValueError("Invalid product identifier")
        for month, quantity in series_history[(product_id, location_id)].items():
            if month.day != 1 or month >= origin:
                raise ValueError("History contains a future or invalid month")
            _finite_nonnegative(quantity, "Historical demand")


def _seasonal_or_ma3(history: Mapping[date, float], target_month: date) -> float:
    """Seasonal forecast with a recursive MA3 fallback from known history."""
    prior_year = shift_months(target_month, -12)
    if prior_year in history:
        return _finite_nonnegative(history[prior_year], "Seasonal forecast")
    return (
        sum(
            _finite_nonnegative(
                history.get(shift_months(target_month, -offset), 0.0), "MA3"
            )
            for offset in range(1, 4)
        )
        / 3.0
    )


def _blended_shares(
    *,
    series_groups: Mapping[SeriesKey, GroupKey],
    series_history: Mapping[SeriesKey, Mapping[date, float]],
    origin: date,
    target_month: date,
) -> dict[SeriesKey, float]:
    """Create an allocation prior with shrinkage from SKU-store to category to store.

    The direct share gets the highest weight when the SKU has sales. For cells
    without sales, category-location and overall store shares prevent known cells
    from becoming structurally impossible in IPF. Weights are set before the
    holdouts and normalized by product.
    """
    products = sorted({product_id for product_id, _location_id in series_groups})
    members_by_product: dict[int, list[SeriesKey]] = defaultdict(list)
    category_by_product: dict[int, str] = {}
    for series, (_location_id, category) in series_groups.items():
        product_id, _location_id = series
        members_by_product[product_id].append(series)
        known_category = category_by_product.setdefault(product_id, category)
        if known_category != category:
            raise ValueError("Product has inconsistent categories across locations")

    recent_months = tuple(shift_months(origin, -offset) for offset in range(1, 13))
    prior_year = shift_months(target_month, -12)
    recent_cell = {
        series: sum(series_history[series].get(month, 0.0) for month in recent_months)
        for series in series_groups
    }
    prior_cell = {
        series: series_history[series].get(prior_year, 0.0) for series in series_groups
    }

    def share_from(
        values: Mapping[SeriesKey, float], members: Sequence[SeriesKey]
    ) -> dict[SeriesKey, float] | None:
        total = sum(values[series] for series in members)
        return (
            {series: values[series] / total for series in members}
            if total > 0
            else None
        )

    direct_recent = {
        product_id: share_from(recent_cell, members_by_product[product_id])
        for product_id in products
    }
    direct_prior = {
        product_id: share_from(prior_cell, members_by_product[product_id])
        for product_id in products
    }

    category_recent: dict[tuple[str, int], float] = defaultdict(float)
    category_prior: dict[tuple[str, int], float] = defaultdict(float)
    location_recent: dict[int, float] = defaultdict(float)
    location_prior: dict[int, float] = defaultdict(float)
    for (product_id, location_id), (_group_location, category) in series_groups.items():
        series = (product_id, location_id)
        category_recent[(category, location_id)] += recent_cell[series]
        category_prior[(category, location_id)] += prior_cell[series]
        location_recent[location_id] += recent_cell[series]
        location_prior[location_id] += prior_cell[series]

    categories = sorted(set(category_by_product.values()))
    locations = sorted({location_id for _product_id, location_id in series_groups})
    category_recent_total = {
        category: sum(
            category_recent[(category, location_id)] for location_id in locations
        )
        for category in categories
    }
    category_prior_total = {
        category: sum(
            category_prior[(category, location_id)] for location_id in locations
        )
        for category in categories
    }
    location_recent_total = sum(location_recent.values())
    location_prior_total = sum(location_prior.values())

    raw_weights: dict[SeriesKey, float] = {}
    for product_id, members in members_by_product.items():
        category = category_by_product[product_id]
        for series in members:
            _product_id, location_id = series
            direct_sources = [
                values[series]
                for values in (direct_recent[product_id], direct_prior[product_id])
                if values is not None
            ]
            direct = (
                sum(direct_sources) / len(direct_sources) if direct_sources else None
            )
            category_sources = [
                source / total
                for source, total in (
                    (
                        category_recent[(category, location_id)],
                        category_recent_total[category],
                    ),
                    (
                        category_prior[(category, location_id)],
                        category_prior_total[category],
                    ),
                )
                if total > 0
            ]
            category_share = (
                sum(category_sources) / len(category_sources)
                if category_sources
                else 0.0
            )
            location_sources = [
                source / total
                for source, total in (
                    (location_recent[location_id], location_recent_total),
                    (location_prior[location_id], location_prior_total),
                )
                if total > 0
            ]
            location_share = (
                sum(location_sources) / len(location_sources)
                if location_sources
                else 0.0
            )
            if direct is None:
                raw_weights[series] = 0.70 * category_share + 0.30 * location_share
            else:
                raw_weights[series] = (
                    0.70 * direct + 0.20 * category_share + 0.10 * location_share
                )

    shares: dict[SeriesKey, float] = {}
    for product_id, members in members_by_product.items():
        total = sum(raw_weights[series] for series in members)
        if total <= 0:
            shares.update({series: 1.0 / len(members) for series in members})
        else:
            shares.update(
                {
                    series: max(raw_weights[series] / total, _EPSILON)
                    for series in members
                }
            )
    return shares


def reconcile_product_location_matrix(
    *,
    initial_weights: Mapping[SeriesKey, float],
    product_totals: Mapping[int, float],
    location_totals: Mapping[int, float],
    max_iterations: int = 500,
    tolerance: float = 1e-8,
) -> dict[SeriesKey, float]:
    """Apply IPF/RAS while preserving product and location totals."""
    if not initial_weights or max_iterations <= 0 or tolerance <= 0:
        raise ValueError("The initial matrix, iteration count, and tolerance must be valid")
    products = set(product_totals)
    locations = set(location_totals)
    if products != {product_id for product_id, _location_id in initial_weights}:
        raise ValueError("Product totals do not cover the initial matrix")
    if locations != {location_id for _product_id, location_id in initial_weights}:
        raise ValueError("Location totals do not cover the initial matrix")
    product_margin = {
        product_id: _finite_nonnegative(total, "Product total")
        for product_id, total in product_totals.items()
    }
    location_margin = {
        location_id: _finite_nonnegative(total, "Location total")
        for location_id, total in location_totals.items()
    }
    if not math.isclose(
        sum(product_margin.values()),
        sum(location_margin.values()),
        rel_tol=tolerance,
        abs_tol=tolerance,
    ):
        raise ValueError("Product and location margins do not have the same total")

    members_by_product: dict[int, list[SeriesKey]] = defaultdict(list)
    members_by_location: dict[int, list[SeriesKey]] = defaultdict(list)
    values: dict[SeriesKey, float] = {}
    for series, weight in initial_weights.items():
        product_id, location_id = series
        members_by_product[product_id].append(series)
        members_by_location[location_id].append(series)
        values[series] = max(_finite_nonnegative(weight, "Peso inicial"), _EPSILON)

    for product_id, total in product_margin.items():
        if total > 0 and not members_by_product[product_id]:
            raise ValueError("Positive product has no cell in the matrix")
    for location_id, total in location_margin.items():
        if total > 0 and not members_by_location[location_id]:
            raise ValueError("Positive location has no cell in the matrix")

    for _iteration in range(max_iterations):
        for product_id, members in members_by_product.items():
            current_total = sum(values[series] for series in members)
            target_total = product_margin[product_id]
            scale = target_total / current_total if current_total > 0 else 0.0
            for series in members:
                values[series] *= scale
        for location_id, members in members_by_location.items():
            current_total = sum(values[series] for series in members)
            target_total = location_margin[location_id]
            scale = target_total / current_total if current_total > 0 else 0.0
            for series in members:
                values[series] *= scale

        margin_error = max(
            [
                abs(
                    sum(values[series] for series in members)
                    - product_margin[product_id]
                )
                for product_id, members in members_by_product.items()
            ]
            + [
                abs(
                    sum(values[series] for series in members)
                    - location_margin[location_id]
                )
                for location_id, members in members_by_location.items()
            ]
        )
        if margin_error <= tolerance:
            return values
    raise RuntimeError("IPF did not converge to the requested margins")


def build_reconciled_allocation_forecasts(
    *,
    series_groups: Mapping[SeriesKey, GroupKey],
    series_history: Mapping[SeriesKey, Mapping[date, float]],
    target_months: Sequence[date],
) -> dict[ForecastKey, float]:
    """Forecast the full SKU-location matrix and reconcile both margins.

    The location margin is the volume anchor because this layer is already
    validated for planning. The product margin gets its own seasonal forecast and
    is rescaled to that total before IPF. This ensures each month reconciles exactly
    in the forecast total by location while preserving the products' relative profile.
    """
    targets = _validate_target_months(target_months)
    origin = targets[0]
    _validate_history(series_groups, series_history, origin)

    product_history: dict[int, dict[date, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    location_history: dict[int, dict[date, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for (product_id, location_id), history in series_history.items():
        for month, quantity in history.items():
            product_history[product_id][month] += float(quantity)
            location_history[location_id][month] += float(quantity)

    forecasts: dict[ForecastKey, float] = {}
    for target_month in targets:
        location_totals = {
            location_id: _seasonal_or_ma3(history, target_month)
            for location_id, history in location_history.items()
        }
        product_totals = {
            product_id: _seasonal_or_ma3(history, target_month)
            for product_id, history in product_history.items()
        }
        location_mass = sum(location_totals.values())
        product_mass = sum(product_totals.values())
        if location_mass > 0 and product_mass > 0:
            scale = location_mass / product_mass
            product_totals = {
                product_id: total * scale
                for product_id, total in product_totals.items()
            }
        elif location_mass > 0:
            historical_product_mass = {
                product_id: sum(history.values())
                for product_id, history in product_history.items()
            }
            total_history = sum(historical_product_mass.values())
            if total_history <= 0:
                product_totals = {
                    product_id: location_mass / len(product_history)
                    for product_id in product_history
                }
            else:
                product_totals = {
                    product_id: location_mass * quantity / total_history
                    for product_id, quantity in historical_product_mass.items()
                }
        else:
            product_totals = {product_id: 0.0 for product_id in product_history}

        shares = _blended_shares(
            series_groups=series_groups,
            series_history=series_history,
            origin=origin,
            target_month=target_month,
        )
        reconciled = reconcile_product_location_matrix(
            initial_weights={
                series: shares[series] * product_totals[series[0]] for series in shares
            },
            product_totals=product_totals,
            location_totals=location_totals,
        )
        for (product_id, location_id), quantity in reconciled.items():
            forecasts[(target_month, product_id, location_id)] = quantity
            product_history[product_id][target_month] = (
                product_history[product_id].get(target_month, 0.0) + quantity
            )
            location_history[location_id][target_month] = (
                location_history[location_id].get(target_month, 0.0) + quantity
            )
    return forecasts
