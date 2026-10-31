# Databricks notebook source
# ruff: noqa: E402

# COMMAND ----------

"""Ranks macro-regions as candidates for a new distribution center (DC).

The company currently operates DCs in the South, North, and Northeast. The
Central-West and Southeast are served by another region and are compared as
candidates for a new DC. Realized demand and forecasts by macro-region use the
`dim_customer.state` to macro-region to DC proxy.

The regional forecast allocates the company-wide total in proportion to each
region's historical share of realized demand. Therefore, `growth_pct` is the same
for all five macro-regions, while `growth_abs_qty` differentiates additional
forecast volume. Candidate ranking includes only regions without their own DC."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
from ml.demand_forecasting.src.distribution_centers import (
    MACRO_REGION_TO_DISTRIBUTION_CENTER,
    STATE_TO_MACRO_REGION,
    macro_regions_with_dedicated_dc,
    macro_regions_without_dedicated_dc,
)
from ml.demand_forecasting.src.modeling import shift_months

# COMMAND ----------


def show_result(df, rows: int = 100) -> None:
    """Display a DataFrame in Databricks, with a local fallback."""
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(rows, truncate=False)


MONTHLY_DEMAND_TABLE = fq_table(
    INTERMEDIATE_SCHEMA,
    "int_monthly_product_location_demand",
)
FORECAST_TABLE = fq_table(MARTS_SCHEMA, "agg_location_demand_forecast")
RECOMMENDATION_TABLE = fq_table(MARTS_SCHEMA, "agg_distribution_center_recommendation")
FCT_SALES_TABLE = fq_table(MARTS_SCHEMA, "fct_sales")
DIM_CUSTOMER_TABLE = fq_table(MARTS_SCHEMA, "dim_customer")
DIM_LOCATION_TABLE = fq_table(MARTS_SCHEMA, "dim_location")
FORECAST_RUN_ID = upstream_task_value(
    "location_aggregate_forecast", "snapshot_id", FORECAST_CONFIG.snapshot_id
)
if FORECAST_RUN_ID != FORECAST_CONFIG.snapshot_id:
    raise ValueError("The received snapshot does not match the execution cutoff")

COMPARISON_MONTHS = len(FORECAST_CONFIG.horizon_months)
ACTUAL_END_DATE = FORECAST_CONFIG.cutoff_date
ACTUAL_START_MONTH = shift_months(
    ACTUAL_END_DATE.replace(day=1),
    -(COMPARISON_MONTHS - 1),
)

DEDICATED_REGIONS = macro_regions_with_dedicated_dc()
CANDIDATE_REGIONS = macro_regions_without_dedicated_dc()

print("Candidate distribution center region ranking parameters")
print(f"  Monthly demand: {MONTHLY_DEMAND_TABLE}")
print(f"  Sales (proxy by customer state): {FCT_SALES_TABLE}")
print(f"  Forecast: {FORECAST_TABLE}")
print(f"  Output: {RECOMMENDATION_TABLE}")
print(f"  Regions with a dedicated DC: {sorted(DEDICATED_REGIONS)}")
print(f"  Candidate regions (without a dedicated DC): {CANDIDATE_REGIONS}")
print(f"  Actuals: {ACTUAL_START_MONTH} to {ACTUAL_END_DATE}")
print(f"  Forecast horizons: 1 to {COMPARISON_MONTHS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Select the snapshot produced in this run
# MAGIC
# MAGIC Same logic as `06_cd_forecast_growth.py`: use the propagated `snapshot_id`
# MAGIC and validate all three horizons before ranking.

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
# MAGIC ### 1.1 Completeness by series, not just by run
# MAGIC
# MAGIC The filter above confirms only that the `run_id` contains horizons 1, 2,
# MAGIC and 3 in aggregate; it does not confirm that each individual series
# MAGIC (`product_variant_id` + `location_id`) contains all three. A series with a
# MAGIC missing horizon would silently distort `total_forecast_qty` (the sum across
# MAGIC all series). Count incomplete series and exclude them before allocating by
# MAGIC macro-region. This is the same adjustment made in `06_cd_forecast_growth.py`.

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
    "(excluded from the forecast total used for regional allocation)"
)

complete_location_df = location_horizon_check_df.filter(location_is_complete).select(
    "location_id"
)
selected_forecast_df = selected_forecast_df.join(
    complete_location_df, ["location_id"], "inner"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Actual demand by macro-region (proxy: customer state)
# MAGIC
# MAGIC As in `06_cd_forecast_growth.py`, but grouped by `macro_region` instead of
# MAGIC `distribution_center_location_id`. This lets the South distribution center
# MAGIC be compared across the three regions it serves.

# COMMAND ----------

state_region_df = spark.createDataFrame(  # noqa: F821
    [{"state": s, "macro_region": r} for s, r in STATE_TO_MACRO_REGION.items()]
)

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
    sales_with_customer_state_df.join(
        state_region_df.select("state"), "state", "left_anti"
    )
    .agg(F.sum("quantity_qty").alias("qty"))
    .collect()[0]["qty"]
    or 0.0
)
print(
    "Quantity excluded because the state is null or unmapped in the actual period: "
    f"{unmapped_state_qty} (see README section 9.3; expected to be 0 in the current dataset)"
)

actual_by_region_df = (
    sales_with_customer_state_df.join(state_region_df, "state", "inner")
    .groupBy("macro_region")
    .agg(F.sum(F.col("quantity_qty").cast("double")).alias("actual_last_3m_demand_qty"))
)

total_actual_qty = actual_by_region_df.agg(
    F.sum("actual_last_3m_demand_qty").alias("total")
).collect()[0]["total"]
if not total_actual_qty:
    raise ValueError("Total actual demand by region is zero or null")

region_count = actual_by_region_df.count()
if region_count != 5:
    raise ValueError(f"Expected 5 macro-regions, found {region_count}")

print(f"Total actual demand (via proxy, 3 months): {total_actual_qty}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Forecast by macro-region: proportional allocation
# MAGIC
# MAGIC Use the same method as `06_cd_forecast_growth.py`, one level lower: allocate
# MAGIC the company-wide forecast total across the five macro-regions in proportion
# MAGIC to each region's actual demand from the proxy.

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

forecast_by_region_df = actual_by_region_df.withColumn(
    "forecast_next_3m_demand_qty",
    F.col("actual_last_3m_demand_qty")
    / F.lit(total_actual_qty)
    * F.lit(total_forecast_qty),
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Growth and ranking among candidate regions
# MAGIC
# MAGIC `candidate_rank` is calculated only for regions without a dedicated DC
# MAGIC (Central-West and Southeast). The South, North, and Northeast already have
# MAGIC a DC and receive a null `candidate_rank`, but remain in the table for audit
# MAGIC and comparison.

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
    forecast_by_region_df.withColumn(
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
    .withColumn(
        "has_dedicated_dc",
        F.col("macro_region").isin(list(DEDICATED_REGIONS)),
    )
    .withColumn(
        "currently_covered_by_location_id",
        F.lit(None).cast("integer"),
    )
)

for region, dc_location_id in MACRO_REGION_TO_DISTRIBUTION_CENTER.items():
    growth_base_df = growth_base_df.withColumn(
        "currently_covered_by_location_id",
        F.when(F.col("macro_region") == F.lit(region), F.lit(dc_location_id)).otherwise(
            F.col("currently_covered_by_location_id")
        ),
    )

dc_identity_df = (
    spark.table(DIM_LOCATION_TABLE)  # noqa: F821
    .filter(F.col("location_type") == F.lit("warehouse"))
    .select(
        F.col("location_id").alias("currently_covered_by_location_id"),
        F.col("location_name").alias("currently_covered_by_name"),
    )
)

# The aggregate forecast is seasonal. Change versus the three immediately prior
# months is useful for capacity planning, but does not measure structural growth.
# Therefore, the ranking below prioritizes monitoring by forecast volume; it is
# never an automatic decision to open a new center.
candidate_rank_window = Window.orderBy(F.col("forecast_next_3m_demand_qty").desc())
created_at = datetime.now(timezone.utc).replace(tzinfo=None)

# `F.rank().over(window)` inside `F.when(...)` does not restrict the window to
# rows that satisfy the condition; it ranks all five macro-regions and only then
# decides whether to keep the value or set it to null. To rank only candidate
# regions, run the window over a DataFrame already filtered to those regions and
# join the result back. Regions with a dedicated DC will not appear in the join
# and will retain a null `candidate_rank`.
candidate_rank_df = (
    growth_base_df.filter(~F.col("has_dedicated_dc"))
    .withColumn("candidate_rank", F.rank().over(candidate_rank_window))
    .select("macro_region", "candidate_rank")
)

recommendation_df = (
    growth_base_df.join(dc_identity_df, "currently_covered_by_location_id", "left")
    .join(candidate_rank_df, "macro_region", "left")
    .select(
        "macro_region",
        "has_dedicated_dc",
        "currently_covered_by_location_id",
        "currently_covered_by_name",
        "actual_last_3m_demand_qty",
        "forecast_next_3m_demand_qty",
        "growth_abs_qty",
        "growth_pct",
        "growth_base_status",
        "candidate_rank",
        F.col("macro_region").alias("candidate_location"),
        F.lit(FORECAST_RUN_ID).alias("source_forecast_run_id"),
        F.lit(ACTUAL_START_MONTH).cast("date").alias("actual_start_month"),
        F.lit(ACTUAL_END_DATE).cast("date").alias("actual_end_date"),
        F.lit(realized_month_count).cast("long").alias("actual_month_count"),
        F.lit(COMPARISON_MONTHS).cast("long").alias("forecast_horizon_count"),
        F.lit(created_at).cast("timestamp").alias("created_at"),
    )
)

candidate_max_forecast = (
    recommendation_df.filter(F.col("candidate_rank").isNotNull())
    .agg(F.max("forecast_next_3m_demand_qty").alias("max_forecast"))
    .collect()[0]["max_forecast"]
)
if candidate_max_forecast is None:
    raise ValueError("No candidate region has a calculable forecast")

score_denominator = float(candidate_max_forecast) or 1.0
recommendation_df = (
    recommendation_df.withColumn(
        "recommendation_score",
        F.when(
            F.col("candidate_rank").isNotNull(),
            F.col("forecast_next_3m_demand_qty") / F.lit(score_denominator),
        ).otherwise(F.lit(None).cast("double")),
    )
    .withColumn(
        "technical_recommendation",
        F.when(
            F.col("candidate_rank").isNotNull(),
            F.lit("do not open a new distribution center now"),
        ).otherwise(F.lit(None).cast("string")),
    )
    .withColumn(
        "recommendation_reason",
        F.when(
            F.col("candidate_rank") == F.lit(1),
            F.lit(
                "Highest forecast volume among regions without a dedicated DC, used "
                "only as a monitoring priority. The increase versus the previous "
                "three months is seasonal and is not evidence of structural growth. "
                "Validate service levels and costs before reconsidering an opening.",
            ),
        )
        .when(
            F.col("candidate_rank").isNotNull(),
            F.lit(
                "Forecast volume is below the monitoring priority; keep current "
                "coverage. The increase versus the previous three months is seasonal "
                "and does not justify opening a distribution center.",
            ),
        )
        .otherwise(F.lit(None).cast("string")),
    )
)
recommendation_count = recommendation_df.count()
if recommendation_count != 5:
    raise ValueError(
        f"Expected 5 macro-regions in the result, found {recommendation_count}"
    )

candidate_count = recommendation_df.filter(F.col("candidate_rank").isNotNull()).count()
if candidate_count != len(CANDIDATE_REGIONS):
    raise ValueError(
        f"Expected {len(CANDIDATE_REGIONS)} ranked candidate regions, "
        f"found {candidate_count}"
    )

show_result(
    recommendation_df.orderBy(
        F.col("candidate_rank").asc_nulls_last(), F.col("macro_region")
    )
)
print(f"Macro-regions evaluated: {recommendation_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Persistence
# MAGIC
# MAGIC This is a new table (`agg_distribution_center_recommendation`) at the
# MAGIC macro-region grain, unlike `agg_location_forecast_growth` (at the
# MAGIC `location_id` grain). Reusing that schema would not make sense because
# MAGIC candidate regions do not have a `location_id`; they do not have a DC yet.

# COMMAND ----------


# The table may have been created on the first run before recommendation fields
# existed. Add only missing columns to keep the MERGE idempotent.

if spark.catalog.tableExists(RECOMMENDATION_TABLE):  # noqa: F821
    existing_columns = {
        field.name
        for field in spark.table(RECOMMENDATION_TABLE).schema  # noqa: F821
    }

    columns_to_add = {
        "candidate_location": "STRING",
        "recommendation_score": "DOUBLE",
        "technical_recommendation": "STRING",
        "recommendation_reason": "STRING",
    }

    missing_columns = [
        f"{name} {data_type}"
        for name, data_type in columns_to_add.items()
        if name not in existing_columns
    ]

    if missing_columns:
        spark.sql(  # noqa: F821
            f"ALTER TABLE {RECOMMENDATION_TABLE} ADD COLUMNS "
            f"({', '.join(missing_columns)})"
        )


if spark.catalog.tableExists(RECOMMENDATION_TABLE):  # noqa: F821
    from delta.tables import DeltaTable

    target = DeltaTable.forName(spark, RECOMMENDATION_TABLE)  # noqa: F821
    (
        target.alias("target")
        .merge(
            recommendation_df.alias("source"),
            "target.macro_region = source.macro_region",
        )
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
else:
    recommendation_df.write.format("delta").mode("overwrite").saveAsTable(
        RECOMMENDATION_TABLE
    )

persisted_df = spark.table(RECOMMENDATION_TABLE)  # noqa: F821
show_result(
    persisted_df.select(
        "macro_region",
        "has_dedicated_dc",
        "currently_covered_by_name",
        "actual_last_3m_demand_qty",
        "forecast_next_3m_demand_qty",
        "growth_abs_qty",
        "candidate_rank",
        "recommendation_score",
        "technical_recommendation",
        "recommendation_reason",
    ).orderBy(F.col("candidate_rank").asc_nulls_last(), F.col("macro_region"))
)

top_candidate = (
    persisted_df.filter(F.col("candidate_rank") == F.lit(1))
    .select(
        "macro_region",
        "growth_abs_qty",
        "technical_recommendation",
        "recommendation_reason",
    )
    .collect()[0]
)
print(
    "Technical recommendation for a new distribution center: "
    f"{top_candidate['macro_region']} - "
    f"{top_candidate['technical_recommendation']} "
    f"(+{top_candidate['growth_abs_qty']:.2f} unidades). "
        f"Reason: {top_candidate['recommendation_reason']}"
)
print(f"Ranking persisted to: {RECOMMENDATION_TABLE}")
print(f"Source forecast run: {FORECAST_RUN_ID}")
print("COMPLETE")
