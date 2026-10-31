# Databricks notebook source
# ruff: noqa: E402

# COMMAND ----------

"""Calculates and persists forecast growth by distribution center (DC).

Order location attribution is unreliable in this dataset: `location_id` does not
necessarily identify the responsible DC. DC demand therefore uses the proxy
`dim_customer.state` to macro-region to DC, calculated from `fct_sales` and
`dim_customer` for the latest three closed months.

There is no forecast trained by customer state. The DC forecast is obtained by
allocating the company-wide forecast (`fct_demand_forecast`) in proportion to each
DC's historical share of realized demand through the same proxy. This assumption
keeps each region's historical share of total forecast growth; the output records
this limitation.

Because the allocation is proportional, `growth_pct` is the same for all three DCs
and represents company-wide aggregate growth. `growth_abs_qty` differentiates DCs
by additional forecast volume and is used for ranking."""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

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
    upstream_task_value,
)
from ml.demand_forecasting.src.distribution_centers import build_state_coverage
from ml.demand_forecasting.src.modeling import shift_months

# COMMAND ----------


def show_result(df: DataFrame, rows: int = 100) -> None:
    """Displays a DataFrame in Databricks, with a local fallback."""
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(rows, truncate=False)


MONTHLY_DEMAND_TABLE = fq_table(
    INTERMEDIATE_SCHEMA,
    "int_monthly_product_location_demand",
)
FORECAST_TABLE = fq_table(MARTS_SCHEMA, "agg_location_demand_forecast")
LOCATION_GROWTH_TABLE = fq_table(MARTS_SCHEMA, "agg_location_forecast_growth")
COVERAGE_TABLE = fq_table(MARTS_SCHEMA, "dim_distribution_center_coverage")
FCT_SALES_TABLE = fq_table(MARTS_SCHEMA, "fct_sales")
DIM_CUSTOMER_TABLE = fq_table(MARTS_SCHEMA, "dim_customer")
DIM_LOCATION_TABLE = fq_table(MARTS_SCHEMA, "dim_location")
FORECAST_RUN_ID = upstream_task_value(
    "location_aggregate_forecast", "snapshot_id", FORECAST_CONFIG.snapshot_id
)
if FORECAST_RUN_ID != FORECAST_CONFIG.snapshot_id:
    raise ValueError("The received snapshot does not match the execution cutoff")

COMPARISON_TYPE = "distribution_center"
COMPARISON_MONTHS = len(FORECAST_CONFIG.horizon_months)
ACTUAL_END_DATE = FORECAST_CONFIG.cutoff_date
ACTUAL_START_MONTH = shift_months(
    ACTUAL_END_DATE.replace(day=1),
    -(COMPARISON_MONTHS - 1),
)
FORECAST_START_MONTH = date.fromisoformat(f"{FORECAST_CONFIG.horizon_months[0]}-01")
FORECAST_END_MONTH = date.fromisoformat(f"{FORECAST_CONFIG.horizon_months[-1]}-01")

print("Distribution center growth parameters")
print(f"  Monthly demand: {MONTHLY_DEMAND_TABLE}")
print(f"  Sales (proxy by customer state): {FCT_SALES_TABLE}")
print(f"  Forecast: {FORECAST_TABLE}")
print(f"  State-to-DC coverage: {COVERAGE_TABLE}")
print(f"  Output: {LOCATION_GROWTH_TABLE} (comparison_type='{COMPARISON_TYPE}')")
print(f"  Actuals: {ACTUAL_START_MONTH} to {ACTUAL_END_DATE}")
print(f"  Forecast: {FORECAST_START_MONTH} to {FORECAST_END_MONTH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Persist state-to-DC coverage (allocation rule evidence)
# MAGIC
# MAGIC The coverage table keeps the allocation rule auditable. It is the source
# MAGIC of truth: anyone can check exactly which states feed each distribution
# MAGIC center without reading this notebook.

# COMMAND ----------

coverage_rows = [
    {
        "state": row.state,
        "macro_region": row.macro_region,
        "distribution_center_location_id": row.distribution_center_location_id,
    }
    for row in build_state_coverage()
]
coverage_df = spark.createDataFrame(coverage_rows)  # noqa: F821

dc_identity_df = spark.table(DIM_LOCATION_TABLE).filter(  # noqa: F821
    F.col("location_type") == F.lit("warehouse")
)
coverage_with_dc_name_df = coverage_df.join(
    dc_identity_df.select(
        F.col("location_id").alias("distribution_center_location_id"),
        F.col("location_name").alias("distribution_center_name"),
        F.col("city").alias("distribution_center_city"),
        F.col("state").alias("distribution_center_state"),
    ),
    "distribution_center_location_id",
    "left",
)

coverage_count = coverage_with_dc_name_df.count()
if coverage_count != 27:
    raise ValueError(f"Expected coverage for 27 states, found {coverage_count}")

coverage_with_dc_name_df.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(COVERAGE_TABLE)

show_result(coverage_with_dc_name_df.orderBy("macro_region", "state"))
print(f"Coverage persisted to: {COVERAGE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Select the snapshot produced in this run
# MAGIC
# MAGIC Same logic as `04_location_forecast_growth.py`: use the propagated
# MAGIC `snapshot_id` and validate all three horizons before allocation.
# MAGIC This logic is duplicated here instead of being extracted to `src/`; that
# MAGIC refactor would affect another contributor's notebook without their review.
# MAGIC Track it as a future improvement.

# COMMAND ----------

forecast_history_df = spark.table(FORECAST_TABLE)  # noqa: F821

selected_forecast_df = forecast_history_df.filter(
    F.col("run_id") == F.lit(FORECAST_RUN_ID)
)
selected_run_summary = selected_forecast_df.agg(
    F.countDistinct("horizon_month").alias("horizon_count"),
    F.min("horizon_month").alias("min_horizon"),
    F.max("horizon_month").alias("max_horizon"),
).first()

if (
    selected_run_summary is None
    or selected_run_summary["horizon_count"] != COMPARISON_MONTHS
    or selected_run_summary["min_horizon"] != 1
    or selected_run_summary["max_horizon"] != COMPARISON_MONTHS
):
    raise ValueError(f"Snapshot {FORECAST_RUN_ID} does not contain all three horizons")

print(f"Selected forecast run: {FORECAST_RUN_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Completeness by series, not just by run
# MAGIC
# MAGIC The filter above confirms only that the `run_id` contains horizons 1, 2,
# MAGIC and 3 in aggregate; it does not confirm that each individual series
# MAGIC (`product_variant_id` + `location_id`) contains all three. A series with a
# MAGIC missing horizon would silently distort `total_forecast_qty` (the sum across
# MAGIC all series). Count incomplete series and exclude them before allocating to
# MAGIC distribution centers.

# COMMAND ----------

location_horizon_check_df = selected_forecast_df.groupBy("location_id").agg(
    F.countDistinct("horizon_month").alias("location_horizon_count"),
    F.min("horizon_month").alias("location_min_horizon"),
    F.max("horizon_month").alias("location_max_horizon"),
)

location_is_complete = (
    (F.col("location_horizon_count") == F.lit(COMPARISON_MONTHS))
    & (F.col("location_min_horizon") == F.lit(1))
    & (F.col("location_max_horizon") == F.lit(COMPARISON_MONTHS))
)

incomplete_location_count = location_horizon_check_df.filter(
    ~location_is_complete
).count()
print(
    "Locations with incomplete horizons in the selected run: "
    f"{incomplete_location_count} "
    "(excluded from the forecast total used for distribution center allocation)"
)

complete_location_df = location_horizon_check_df.filter(location_is_complete).select(
    "location_id"
)
selected_forecast_df = selected_forecast_df.join(
    complete_location_df, ["location_id"], "inner"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Actual demand by distribution center (proxy: customer state)
# MAGIC
# MAGIC Read directly from `fct_sales` + `dim_customer`, not from
# MAGIC `int_monthly_product_location_demand`; that model has a product + operational
# MAGIC location grain and does not include customer state.

# COMMAND ----------

dim_customer_df = spark.table(DIM_CUSTOMER_TABLE)  # noqa: F821

sales_paid_in_period_df = spark.table(FCT_SALES_TABLE).filter(  # noqa: F821
    (F.col("order_status") == F.lit("paid"))
    & (F.col("sale_datetime").cast("date") >= F.lit(ACTUAL_START_MONTH))
    & (F.col("sale_datetime").cast("date") <= F.lit(ACTUAL_END_DATE))
)

unmatched_customer_qty = (
    sales_paid_in_period_df.join(dim_customer_df, "customer_key", "left_anti")
    .agg(F.sum("quantity_qty").alias("qty"))
    .collect()[0]["qty"]
    or 0.0
)
print(
    "Quantity excluded because the sale has no matching customer in dim_customer "
    f"(inner join): {unmatched_customer_qty} "
    "(expected to be 0 in the current dataset; see README section 9.3)"
)

sales_with_customer_state_df = sales_paid_in_period_df.join(
    dim_customer_df, "customer_key", "inner"
).select("quantity_qty", "state")

unmapped_state_qty = (
    sales_with_customer_state_df.join(coverage_df.select("state"), "state", "left_anti")
    .agg(F.sum("quantity_qty").alias("qty"))
    .collect()[0]["qty"]
    or 0.0
)
print(
    "Quantity excluded because the state is null or unmapped in the actual period: "
    f"{unmapped_state_qty} (see README section 9.3; expected to be 0 in the current dataset)"
)

actual_by_dc_df = (
    sales_with_customer_state_df.join(coverage_df, "state", "inner")
    .groupBy("distribution_center_location_id")
    .agg(F.sum(F.col("quantity_qty").cast("double")).alias("actual_last_3m_demand_qty"))
    .withColumnRenamed(
        "distribution_center_location_id",
        "location_id",
    )
)

total_actual_qty = actual_by_dc_df.agg(
    F.sum("actual_last_3m_demand_qty").alias("total")
).collect()[0]["total"]
if not total_actual_qty:
    raise ValueError("Total actual demand by distribution center is zero or null")

print(f"Total actual demand (via proxy, 3 months): {total_actual_qty}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Distribution center forecast: allocation by historical share
# MAGIC
# MAGIC Allocate the company-wide forecast total (the sum of `forecast_demand_qty`
# MAGIC across all locations in the selected run) among the three distribution
# MAGIC centers in proportion to each center's actual demand from the proxy.

# COMMAND ----------

total_forecast_qty = (
    selected_forecast_df.filter(F.col("horizon_month").between(1, COMPARISON_MONTHS))
    .agg(F.sum(F.col("forecast_demand_qty").cast("double")).alias("total"))
    .collect()[0]["total"]
)
if not total_forecast_qty:
    raise ValueError("Company-wide forecast total is zero or null")

print(
    f"Company-wide forecast total (next {COMPARISON_MONTHS} months): {total_forecast_qty}"
)

forecast_by_dc_df = actual_by_dc_df.withColumn(
    "forecast_next_3m_demand_qty",
    F.col("actual_last_3m_demand_qty")
    / F.lit(total_actual_qty)
    * F.lit(total_forecast_qty),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Distribution center growth, ranking, and identity
# MAGIC
# MAGIC Apply the same status rules as `04_location_forecast_growth.py`
# MAGIC (`insufficient_actual_history`, `new_or_inactive_base`), with one adaptation:
# MAGIC count months from `int_monthly_product_location_demand` for the period, not
# MAGIC from the proxy (the proxy is not monthly at the row level).

# COMMAND ----------

realized_month_count = (
    spark.table(MONTHLY_DEMAND_TABLE)  # noqa: F821
    .filter(
        (F.col("demand_month") >= F.lit(ACTUAL_START_MONTH))
        & (F.col("demand_month") <= F.lit(ACTUAL_END_DATE))
    )
    .agg(F.countDistinct("demand_month").alias("months"))
    .collect()[0]["months"]
)

growth_base_df = (
    forecast_by_dc_df.withColumn(
        "growth_abs_qty",
        F.col("forecast_next_3m_demand_qty") - F.col("actual_last_3m_demand_qty"),
    )
    .withColumn(
        "growth_pct",
        F.when(
            (F.lit(realized_month_count) == F.lit(COMPARISON_MONTHS))
            & (F.col("actual_last_3m_demand_qty") > F.lit(0.0)),
            F.col("growth_abs_qty") / F.col("actual_last_3m_demand_qty") * F.lit(100.0),
        ).otherwise(F.lit(None).cast("double")),
    )
    .withColumn(
        "growth_base_status",
        F.when(
            F.lit(realized_month_count) < F.lit(COMPARISON_MONTHS),
            F.lit("insufficient_actual_history"),
        )
        .when(
            F.col("actual_last_3m_demand_qty") == F.lit(0.0),
            F.lit("new_or_inactive_base"),
        )
        .otherwise(F.lit("comparable")),
    )
)

growth_rank_abs_window = Window.orderBy(F.col("growth_abs_qty").desc())
growth_rank_pct_window = Window.orderBy(F.col("growth_pct").desc_nulls_last())
created_at = datetime.now(timezone.utc).replace(tzinfo=None)

dc_identity_for_join_df = dc_identity_df.select(
    "location_id",
    "location_key",
    "location_name",
    "location_type",
    "city",
    "state",
)

cd_growth_df = (
    growth_base_df.join(dc_identity_for_join_df, "location_id", "inner")
    .withColumn("growth_rank_abs", F.rank().over(growth_rank_abs_window))
    .withColumn(
        "growth_rank_pct",
        F.when(
            F.col("growth_pct").isNotNull(),
            F.rank().over(growth_rank_pct_window),
        ).otherwise(F.lit(None).cast("integer")),
    )
    .select(
        "location_id",
        "location_key",
        "location_name",
        "location_type",
        "city",
        "state",
        F.lit(COMPARISON_TYPE).alias("comparison_type"),
        F.lit(FORECAST_RUN_ID).alias("source_forecast_run_id"),
        F.lit(ACTUAL_START_MONTH).cast("date").alias("actual_start_month"),
        F.lit(ACTUAL_END_DATE).cast("date").alias("actual_end_date"),
        F.lit(FORECAST_START_MONTH).cast("date").alias("forecast_start_month"),
        F.lit(FORECAST_END_MONTH).cast("date").alias("forecast_end_month"),
        "actual_last_3m_demand_qty",
        "forecast_next_3m_demand_qty",
        "growth_abs_qty",
        "growth_pct",
        "growth_rank_abs",
        "growth_rank_pct",
        "growth_base_status",
        F.lit(realized_month_count).cast("long").alias("actual_month_count"),
        F.lit(COMPARISON_MONTHS).cast("long").alias("forecast_horizon_count"),
        F.lit(created_at).cast("timestamp").alias("created_at"),
    )
)

dc_count = cd_growth_df.count()
if dc_count != 3:
    raise ValueError(f"Expected 3 distribution centers in the ranking, found {dc_count}")

show_result(cd_growth_df.orderBy("growth_rank_abs", "location_id"))
print(f"Ranked distribution centers: {dc_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Persist to the same growth aggregate
# MAGIC
# MAGIC Reuse `agg_location_forecast_growth` (the same schema as
# MAGIC `04_location_forecast_growth.py`) with `comparison_type =
# MAGIC 'distribution_center'` instead of creating a new table. The
# MAGIC `comparison_type` column supports multiple comparison types, and the store
# MAGIC `MERGE` already preserves other types explicitly.
# MAGIC
# MAGIC The forecast interval is the global horizon used for proportional
# MAGIC allocation. It does not represent a model trained for each distribution
# MAGIC center, but it makes the period compared in the ranking explicit.

# COMMAND ----------

if spark.catalog.tableExists(LOCATION_GROWTH_TABLE):  # noqa: F821
    from delta.tables import DeltaTable

    target = DeltaTable.forName(spark, LOCATION_GROWTH_TABLE)  # noqa: F821
    merge_condition = (
        "target.location_id = source.location_id "
        "AND target.comparison_type = source.comparison_type"
    )
    (
        target.alias("target")
        .merge(cd_growth_df.alias("source"), merge_condition)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .whenNotMatchedBySourceDelete(
            condition=f"target.comparison_type = '{COMPARISON_TYPE}'"
        )
        .execute()
    )
else:
    cd_growth_df.write.format("delta").mode("append").saveAsTable(LOCATION_GROWTH_TABLE)

persisted_growth_df = spark.table(LOCATION_GROWTH_TABLE).filter(  # noqa: F821
    F.col("comparison_type") == F.lit(COMPARISON_TYPE)
)
show_result(
    persisted_growth_df.select(
        "location_name",
        "location_type",
        "actual_last_3m_demand_qty",
        "forecast_next_3m_demand_qty",
        "growth_abs_qty",
        "growth_pct",
        "growth_rank_abs",
    ).orderBy("growth_rank_abs", "location_id")
)

print(
    f"Ranking persisted to: {LOCATION_GROWTH_TABLE} "
    f"(comparison_type='{COMPARISON_TYPE}')"
)
print(f"Source forecast run: {FORECAST_RUN_ID}")
print("COMPLETE")
