"""Thresholds and classifications for the product seasonality ranking.

The single source of truth for thresholds documented in the README
(section 8.3): eligibility, peak/neutral/low seasonal-index classification,
and confidence based on observations per month. New notebooks should import
these values instead of repeating them, following the same pattern as
`config.py` for regression eligibility (README sections 3 and 4.1).

This module performs no I/O and does not depend on Spark. The
`notebooks/05_seasonality_ranking.py` notebook reads the monthly dataset,
aggregates it in Spark, and uses these constants/functions in its
`when`/`otherwise` expressions."""

from __future__ import annotations

__all__ = [
    "SEASONALITY_MIN_HISTORY_MONTHS",
    "SEASONALITY_MIN_POSITIVE_MONTHS",
    "SEASONAL_PEAK_INDEX_THRESHOLD",
    "SEASONAL_LOW_INDEX_THRESHOLD",
    "SEASONALITY_CLASS_PEAK",
    "SEASONALITY_CLASS_NEUTRAL",
    "SEASONALITY_CLASS_LOW",
    "HIGH_CONFIDENCE_MIN_OBSERVATIONS",
    "MEDIUM_CONFIDENCE_MIN_OBSERVATIONS",
    "CONFIDENCE_HIGH",
    "CONFIDENCE_MEDIUM",
    "CONFIDENCE_LOW",
    "LOCATION_TYPE_STORE",
    "LOCATION_TYPE_WAREHOUSE",
    "ANALYSIS_LOCATION_TYPES",
    "MONTHS_IN_YEAR",
    "classify_seasonality_class",
    "classify_confidence",
]

# Product eligibility for seasonality analysis (aggregating locations). Unlike
# regression eligibility in config.py (6 months of history and 3 positive months
# at the product-location grain), this uses the product grain across locations,
# and the window must cover a full annual cycle for the monthly index to be useful.
SEASONALITY_MIN_HISTORY_MONTHS = 12
SEASONALITY_MIN_POSITIVE_MONTHS = 6

# Classification of each product-month seasonal index
# (seasonality_index = avg_monthly_demand_qty[m] / overall_avg_monthly_demand_qty).
SEASONAL_PEAK_INDEX_THRESHOLD = 1.20
SEASONAL_LOW_INDEX_THRESHOLD = 0.80

SEASONALITY_CLASS_PEAK = "peak"
SEASONALITY_CLASS_NEUTRAL = "neutral"
SEASONALITY_CLASS_LOW = "low"

# Product confidence based on the minimum number of observed years across the 12
# months (min_obs). This does not exclude products from the ranking; it qualifies CV.
HIGH_CONFIDENCE_MIN_OBSERVATIONS = 3
MEDIUM_CONFIDENCE_MIN_OBSERVATIONS = 2

CONFIDENCE_HIGH = "high"
CONFIDENCE_MEDIUM = "medium"
CONFIDENCE_LOW = "low"

# Location types used in seasonality analysis (README, section 9.1). The explicit
# filter clarifies the scope when reading the monthly dataset, even though these
# are currently the only two location_type values in dim_location.
LOCATION_TYPE_STORE = "store"
LOCATION_TYPE_WAREHOUSE = "warehouse"
ANALYSIS_LOCATION_TYPES = (LOCATION_TYPE_STORE, LOCATION_TYPE_WAREHOUSE)

# Number of months in a year, used to validate that the monthly seasonality table
# (grain product_variant_id + month_number) has exactly one row per month for
# each eligible product. This differs from SEASONALITY_MIN_HISTORY_MONTHS: that
# is the minimum history threshold for eligibility; this is the exact expected
# count after monthly aggregation.
MONTHS_IN_YEAR = 12


def classify_seasonality_class(index: float) -> str:
    """Classify a month as peak, neutral, or low based on its seasonal index.

    `peak` when index >= 1.20, `low` when index <= 0.80, and `neutral` in the
    open interval between the thresholds (the boundary values belong to `peak`
    and `low`, not `neutral`).
    """
    if index >= SEASONAL_PEAK_INDEX_THRESHOLD:
        return SEASONALITY_CLASS_PEAK
    if index <= SEASONAL_LOW_INDEX_THRESHOLD:
        return SEASONALITY_CLASS_LOW
    return SEASONALITY_CLASS_NEUTRAL


def classify_confidence(min_observations: int) -> str:
    """Classify product confidence by the minimum number of observations.

    `high` when min_observations >= 3, `medium` when it is 2, and `low` when it
    is 0 or 1.
    """
    if min_observations >= HIGH_CONFIDENCE_MIN_OBSERVATIONS:
        return CONFIDENCE_HIGH
    if min_observations >= MEDIUM_CONFIDENCE_MIN_OBSERVATIONS:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_LOW
