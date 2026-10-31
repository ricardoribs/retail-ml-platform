# Databricks notebook source
# ruff: noqa: E402, F821

"""Retrospectively validates the store/DC planning forecast.

This notebook does not train a model, change the published snapshot, or write tables.
It compares the snapshot identified by the cutoff with sales realized only during the
three months after the cutoff. Actuals are an evaluation holdout, never a forecast input.

To reproduce validation for September through November 2026, run with the
`reference_date=2026-09-01` widget. This sets the cutoff to 2026-08-31 and selects
snapshot `forecast_2026_08`."""

from __future__ import annotations

import sys
from datetime import date
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
    MARTS_SCHEMA,
    fq_table,
    shift_month_start,
)


def show_result(df: DataFrame, rows: int = 100) -> None:
    """Display a table in either a notebook or a job."""
    try:
        display(df)
    except NameError:
        df.show(rows, truncate=False)


FORECAST_TABLE = fq_table(MARTS_SCHEMA, "agg_location_demand_forecast")
SALES_TABLE = fq_table(MARTS_SCHEMA, "fct_sales")
LOCATION_TABLE = fq_table(MARTS_SCHEMA, "dim_location")
FORECAST_RUN_ID = FORECAST_CONFIG.snapshot_id
CUTOFF_DATE = FORECAST_CONFIG.cutoff_date
FORECAST_MONTHS = tuple(
    date.fromisoformat(f"{month}-01") for month in FORECAST_CONFIG.horizon_months
)
EXPECTED_HORIZONS = len(FORECAST_MONTHS)
ACTUAL_END_EXCLUSIVE = shift_month_start(FORECAST_MONTHS[-1], 1)

if any(month <= CUTOFF_DATE for month in FORECAST_MONTHS):
    raise AssertionError("The evaluation horizon must be after the cutoff")

print("Retrospective validation of store/distribution center planning")
print(f"  Snapshot: {FORECAST_RUN_ID}")
print(f"  Cutoff: {CUTOFF_DATE}")
print(f"  Holdout: {', '.join(FORECAST_CONFIG.horizon_months)}")
print(f"  Published forecast source: {FORECAST_TABLE}")
print(f"  Test actuals source: {SALES_TABLE}")
print("  Persistence: none (read-only evaluation)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Forecast published at the cutoff
# MAGIC
# MAGIC The `run_id` is derived from the cutoff. This notebook does not read the
# MAGIC training matrix, recalculate forecasts, or use actuals after the cutoff to
# MAGIC select a model.

# COMMAND ----------

for table_name in (FORECAST_TABLE, SALES_TABLE, LOCATION_TABLE):
    if not spark.catalog.tableExists(table_name):
        raise ValueError(f"Required table not found: {table_name}")

forecast_df = (
    spark.table(FORECAST_TABLE)
    .filter(F.col("run_id") == F.lit(FORECAST_RUN_ID))
    .filter(F.col("forecast_month").isin(*FORECAST_MONTHS))
    .select(
        "run_id",
        "forecast_month",
        "horizon_month",
        "location_id",
        "location_name",
        "location_type",
        "city",
        "state",
        F.col("forecast_demand_qty").cast("double").alias("forecast_demand_qty"),
        "model_name",
        "model_version",
        "forecast_scope",
    )
)

if forecast_df.select("forecast_month").distinct().count() != EXPECTED_HORIZONS:
    raise AssertionError(
        f"Snapshot {FORECAST_RUN_ID} does not contain all {EXPECTED_HORIZONS} holdout months"
    )
if (
    forecast_df.groupBy("location_id")
    .agg(F.countDistinct("forecast_month").alias("horizon_count"))
    .filter(F.col("horizon_count") != F.lit(EXPECTED_HORIZONS))
    .limit(1)
    .count()
):
    raise AssertionError("A location has an incomplete horizon in the published forecast")
if (
    forecast_df.groupBy("forecast_month", "location_id")
    .count()
    .filter(F.col("count") != F.lit(1))
    .limit(1)
    .count()
):
    raise AssertionError("The published forecast has a duplicate month-location key")
if (
    forecast_df.filter(F.col("forecast_scope") != "location_capacity_planning")
    .limit(1)
    .count()
):
    raise AssertionError("The selected snapshot is not a planning forecast")

expected_cells = (
    forecast_df.select("location_id").distinct().count() * EXPECTED_HORIZONS
)
print(f"Celulas esperadas no holdout: {expected_cells}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Actuals after the cutoff
# MAGIC
# MAGIC Paid sales below are queried only to measure error. They occur after
# MAGIC `CUTOFF_DATE` and therefore are not part of the history used by notebook 20
# MAGIC to publish this snapshot.

# COMMAND ----------

actual_df = (
    spark.table(SALES_TABLE)
    .alias("sales")
    .join(
        spark.table(LOCATION_TABLE).alias("location"),
        F.col("sales.location_key") == F.col("location.location_key"),
        "inner",
    )
    .filter(F.col("sales.order_status") == F.lit(FORECAST_CONFIG.order_status_filter))
    .filter(F.to_date("sales.sale_datetime") > F.lit(CUTOFF_DATE))
    .filter(F.to_date("sales.sale_datetime") < F.lit(ACTUAL_END_EXCLUSIVE))
    .select(
        F.trunc(F.to_date("sales.sale_datetime"), "month").alias("forecast_month"),
        F.col("location.location_id").alias("location_id"),
        F.col("sales.quantity_qty").cast("double").alias("quantity_qty"),
    )
    .filter(F.col("forecast_month").isin(*FORECAST_MONTHS))
    .groupBy("forecast_month", "location_id")
    .agg(F.sum("quantity_qty").alias("actual_demand_qty"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Error at the planning grain and monthly total

# COMMAND ----------

comparison_df = (
    forecast_df.alias("forecast")
    .join(actual_df.alias("actual"), ["forecast_month", "location_id"], "full")
    .withColumn(
        "forecast_demand_qty", F.coalesce(F.col("forecast_demand_qty"), F.lit(0.0))
    )
    .withColumn("actual_demand_qty", F.coalesce(F.col("actual_demand_qty"), F.lit(0.0)))
    .withColumn(
        "absolute_error_qty",
        F.abs(F.col("forecast_demand_qty") - F.col("actual_demand_qty")),
    )
)

if comparison_df.count() != expected_cells:
    raise AssertionError(
        "The actuals do not match the published universe; "
        "check for new locations or missing forecasts"
    )
if comparison_df.filter(F.col("actual_demand_qty") < F.lit(0.0)).limit(1).count():
    raise AssertionError("Test actuals contain negative demand")

monthly_df = (
    comparison_df.groupBy("forecast_month")
    .agg(
        F.sum("forecast_demand_qty").alias("forecast_demand_qty"),
        F.sum("actual_demand_qty").alias("actual_demand_qty"),
    )
    .withColumn(
        "absolute_error_qty",
        F.abs(F.col("forecast_demand_qty") - F.col("actual_demand_qty")),
    )
    .orderBy("forecast_month")
)

location_totals = comparison_df.agg(
    F.sum("absolute_error_qty").alias("absolute_error_qty"),
    F.sum("actual_demand_qty").alias("actual_demand_qty"),
).first()
monthly_totals = monthly_df.agg(
    F.sum("absolute_error_qty").alias("absolute_error_qty"),
    F.sum("actual_demand_qty").alias("actual_demand_qty"),
).first()

if not location_totals["actual_demand_qty"] or not monthly_totals["actual_demand_qty"]:
    raise AssertionError(
        "Cannot calculate WAPE when total actuals are zero"
    )

metrics_df = spark.createDataFrame(
    [
        (
            "monthly_total",
            int(EXPECTED_HORIZONS),
            float(monthly_totals["absolute_error_qty"]),
            float(monthly_totals["actual_demand_qty"]),
            100
            * float(monthly_totals["absolute_error_qty"])
            / float(monthly_totals["actual_demand_qty"]),
        ),
        (
            "month_location",
            int(expected_cells),
            float(location_totals["absolute_error_qty"]),
            float(location_totals["actual_demand_qty"]),
            100
            * float(location_totals["absolute_error_qty"])
            / float(location_totals["actual_demand_qty"]),
        ),
    ],
    [
        "evaluation_grain",
        "evaluated_cells",
        "absolute_error_qty",
        "actual_demand_qty",
        "wape_pct",
    ],
)

show_result(monthly_df)
show_result(
    comparison_df.select(
        "forecast_month",
        "location_id",
        "location_name",
        "location_type",
        "forecast_demand_qty",
        "actual_demand_qty",
        "absolute_error_qty",
    ).orderBy("forecast_month", "location_id")
)
show_result(metrics_df.orderBy("evaluation_grain"))

print("RETROSPECTIVE PLANNING VALIDATION COMPLETE")
