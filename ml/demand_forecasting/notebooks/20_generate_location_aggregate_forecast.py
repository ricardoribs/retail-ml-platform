# Databricks notebook source
# ruff: noqa: E402

"""Publishes the seasonal aggregate forecast by store/DC for planning.

Backtest `19_location_allocation_backtest.py` showed that the seasonal location
baseline reduces WAPE in all three folds at the store/DC level, while automatic
allocation across SKUs does not beat the zero baseline. This output supports capacity
planning and growth rankings; it does not replace or reconcile the product-location
forecast."""

from __future__ import annotations

import sys
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

if TYPE_CHECKING:
    display: Any
    spark: Any

_REPO_ROOT = next(
    (
        candidate
        for start in (Path.cwd(), *(Path(path) for path in sys.path if path))
        for candidate in (start, *start.parents)
        if (candidate / "pyproject.toml").is_file()
        and (candidate / "ml" / "demand_forecasting").is_dir()
    ),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError("Could not locate the project root in Databricks")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.demand_forecasting.src.config import (
    FORECAST_CONFIG,
    INTERMEDIATE_SCHEMA,
    MARTS_SCHEMA,
    fq_table,
    set_task_value,
)
from ml.demand_forecasting.src.location_forecasting import (
    LOCATION_SEASONAL_MODEL_NAME,
    LOCATION_SEASONAL_MODEL_VERSION,
    build_location_seasonal_forecasts,
)
from ml.demand_forecasting.src.modeling import LABEL_COLUMN


def show_result(df: DataFrame, rows: int = 100) -> None:
    """Display a table in either a notebook or a job."""
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(rows, truncate=False)


FEATURE_MATRIX_TABLE = fq_table(INTERMEDIATE_SCHEMA, "int_forecast_feature_matrix")
FORECAST_TABLE = fq_table(MARTS_SCHEMA, "agg_location_demand_forecast")
FORECAST_RUN_ID = FORECAST_CONFIG.snapshot_id
FORECAST_MONTHS = tuple(
    date.fromisoformat(f"{month}-01") for month in FORECAST_CONFIG.horizon_months
)
CUTOFF_MONTH = FORECAST_CONFIG.cutoff_date.replace(day=1)
LOCATION_STATIC_COLUMNS = (
    "location_id",
    "location_key",
    "location_name",
    "location_type",
    "city",
    "state",
)

print("Location-level aggregate forecast parameters")
print(f"  Source: {FEATURE_MATRIX_TABLE}")
print(f"  Output: {FORECAST_TABLE}")
print(f"  Cutoff: {FORECAST_CONFIG.cutoff_date}")
print(f"  Months: {', '.join(FORECAST_CONFIG.horizon_months)}")
print(f"  Modelo: {LOCATION_SEASONAL_MODEL_VERSION}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Location history and identity
# MAGIC
# MAGIC Quantities are summed by store/distribution center only through the cutoff.
# MAGIC Identity is taken from the latest closed month. Any variation within the
# MAGIC same location stops execution rather than publishing an ambiguous ranking.

# COMMAND ----------

history_df = spark.table(FEATURE_MATRIX_TABLE).filter(  # noqa: F821
    F.col("demand_month") <= F.lit(FORECAST_CONFIG.cutoff_date)
)
invalid_demand = history_df.filter(
    F.col(LABEL_COLUMN).isNull() | (F.col(LABEL_COLUMN) < F.lit(0))
).count()
if invalid_demand:
    raise ValueError(f"Ha {invalid_demand} demandas nulas ou negativas")

location_history: dict[int, dict[date, float]] = defaultdict(dict)
for row in (
    history_df.groupBy("demand_month", "location_id")
    .agg(F.sum(F.col(LABEL_COLUMN).cast("double")).alias("demand_qty"))
    .collect()
):
    location_history[row["location_id"]][row["demand_month"]] = float(row["demand_qty"])

latest_location_df = (
    history_df.filter(F.col("demand_month") == F.lit(CUTOFF_MONTH))
    .groupBy(*LOCATION_STATIC_COLUMNS)
    .count()
    .drop("count")
)
duplicate_identity_count = (
    latest_location_df.groupBy("location_id")
    .count()
    .filter(F.col("count") != 1)
    .count()
)
if duplicate_identity_count:
    raise ValueError("Duplicate static identity found for a location")
location_static = {
    row["location_id"]: {column: row[column] for column in LOCATION_STATIC_COLUMNS}
    for row in latest_location_df.collect()
}
if set(location_history) != set(location_static):
    raise ValueError("History and identity do not cover the same locations")

print(f"Locations in scope: {len(location_history)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Recursive seasonal forecast and completeness contracts
# MAGIC
# MAGIC The rule uses the same location and month from the previous year. MA3 is a
# MAGIC recursive fallback used only when a new horizon has no annual equivalent.

# COMMAND ----------

predictions = build_location_seasonal_forecasts(
    location_history=location_history,
    target_months=FORECAST_MONTHS,
)
created_at = datetime.now(timezone.utc).replace(tzinfo=None)
forecast_records = [
    {
        "run_id": FORECAST_RUN_ID,
        "forecast_month": forecast_month,
        "horizon_month": horizon,
        **location_static[location_id],
        "forecast_demand_qty": float(forecast_qty),
        "model_name": LOCATION_SEASONAL_MODEL_NAME,
        "model_version": LOCATION_SEASONAL_MODEL_VERSION,
        "forecast_scope": "location_capacity_planning",
        "created_at": created_at,
    }
    for horizon, forecast_month in enumerate(FORECAST_MONTHS, start=1)
    for location_id in sorted(location_history)
    for forecast_qty in (predictions[(forecast_month, location_id)],)
]
forecast_df = spark.createDataFrame(forecast_records)  # noqa: F821
expected_rows = len(location_history) * len(FORECAST_MONTHS)
if forecast_df.count() != expected_rows:
    raise AssertionError("Aggregate forecast has an unexpected number of rows")
if forecast_df.filter(F.col("forecast_demand_qty") < F.lit(0)).limit(1).count():
    raise AssertionError("Aggregate forecast contains a negative quantity")
show_result(
    forecast_df.groupBy("forecast_month", "location_type")
    .agg(F.sum("forecast_demand_qty").alias("forecast_demand_qty"))
    .orderBy("forecast_month", "location_type")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Idempotent persistence

# COMMAND ----------

if spark.catalog.tableExists(FORECAST_TABLE):  # noqa: F821
    from delta.tables import DeltaTable

    target = DeltaTable.forName(spark, FORECAST_TABLE)  # noqa: F821
    merge_condition = (
        "target.run_id = source.run_id "
        "AND target.forecast_month = source.forecast_month "
        "AND target.location_id = source.location_id"
    )
    (
        target.alias("target")
        .merge(forecast_df.alias("source"), merge_condition)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .whenNotMatchedBySourceDelete(condition=f"target.run_id = '{FORECAST_RUN_ID}'")
        .execute()
    )
else:
    forecast_df.write.format("delta").mode("append").saveAsTable(FORECAST_TABLE)

persisted_df = spark.table(FORECAST_TABLE).filter(  # noqa: F821
    F.col("run_id") == F.lit(FORECAST_RUN_ID)
)
persisted_count = persisted_df.count()
if persisted_count != expected_rows:
    raise AssertionError("Aggregate forecast persistence is incomplete")
incomplete_locations = (
    persisted_df.groupBy("location_id")
    .agg(F.countDistinct("horizon_month").alias("horizon_count"))
    .filter(F.col("horizon_count") != F.lit(len(FORECAST_MONTHS)))
    .count()
)
if incomplete_locations:
    raise AssertionError("Some locations are missing aggregate forecast horizons")

set_task_value("snapshot_id", FORECAST_RUN_ID, required=True)
set_task_value("has_actionable_aggregate_forecast", True, required=True)
print(f"Aggregate forecast persisted to: {FORECAST_TABLE}")
print("Aggregate forecast is actionable for rankings: True")
