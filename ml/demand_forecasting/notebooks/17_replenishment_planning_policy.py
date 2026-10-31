# Databricks notebook source
# ruff: noqa: E402

# COMMAND ----------

"""Publishes the manual replenishment policy by product and location.

The output combines the forecast profile and status with the Gold inventory snapshot.
The decision uses balance, lead time, reorder point, and coverage; it never treats a
zero forecast fallback as future demand or a purchase order."""

from __future__ import annotations

import sys
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
    MARTS_SCHEMA,
    fq_table,
    set_task_value,
    upstream_task_value,
)
from ml.demand_forecasting.src.replenishment_policy import decide_replenishment

# COMMAND ----------


def show_result(df: DataFrame, rows: int = 100) -> None:
    """Displays a DataFrame in Databricks, with a local fallback."""
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(rows, truncate=False)


INVENTORY_TABLE = fq_table(MARTS_SCHEMA, "fct_inventory_replenishment_snapshot")
SERIES_POLICY_TABLE = fq_table(MARTS_SCHEMA, "fct_forecast_series_policy")
PUBLICATION_TABLE = fq_table(MARTS_SCHEMA, "fct_forecast_series_publication")
OUTPUT_TABLE = fq_table(MARTS_SCHEMA, "fct_replenishment_planning_policy")
FORECAST_RUN_ID = upstream_task_value(
    "demand_forecast_generate", "snapshot_id", FORECAST_CONFIG.snapshot_id
)
DBT_REFERENCE_DATE = upstream_task_value(
    "dbt_transformation",
    "reference_date",
    FORECAST_CONFIG.reference_date.isoformat(),
)
try:
    INVENTORY_REFERENCE_DATE = date.fromisoformat(DBT_REFERENCE_DATE)
except ValueError as exc:
    raise ValueError(
        f"Reference date received from dbt is invalid: {DBT_REFERENCE_DATE}"
    ) from exc
POLICY_VERSION = "replenishment_policy_v1"

print("Replenishment policy parameters")
print(f"  Inventory snapshot    : {INVENTORY_TABLE}")
print(f"  Per-series policy     : {SERIES_POLICY_TABLE}")
print(f"  Status de forecast    : {PUBLICATION_TABLE}")
print(f"  Output                : {OUTPUT_TABLE}")
print(f"  Forecast run ID       : {FORECAST_RUN_ID}")
print(f"  Data operacional dbt  : {INVENTORY_REFERENCE_DATE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Sources and snapshot consistency
# MAGIC
# MAGIC Inventory is always read at the latest reference date. All
# MAGIC series must have a published profile and status before a decision is made.

# COMMAND ----------

for table_name in (INVENTORY_TABLE, SERIES_POLICY_TABLE, PUBLICATION_TABLE):
    if not spark.catalog.tableExists(table_name):  # noqa: F821
        raise ValueError(f"Required table not found: {table_name}")

inventory_reference_date = INVENTORY_REFERENCE_DATE

inventory_df = (
    spark.table(INVENTORY_TABLE)  # noqa: F821
    .filter(F.col("reference_date") == F.lit(inventory_reference_date))
    .select(
        "product_variant_id",
        "location_id",
        "current_stock_qty",
        "open_purchase_qty",
        "lead_time_days",
        "effective_reorder_point",
        "stock_coverage_days",
        "is_reorder_needed",
        "is_reorder_needed_after_inbound",
        "is_stockout_risk",
        "suggested_purchase_qty",
    )
)
series_policy_df = (
    spark.table(SERIES_POLICY_TABLE)  # noqa: F821
    .filter(F.col("run_id") == F.lit(FORECAST_RUN_ID))
    .select(
        "product_variant_id",
        "location_id",
        "demand_segment",
        "confidence_level",
    )
)
publication_df = (
    spark.table(PUBLICATION_TABLE)  # noqa: F821
    .filter(F.col("run_id") == F.lit(FORECAST_RUN_ID))
    .select(
        "product_variant_id",
        "location_id",
        "publication_status",
        "review_required",
        "automation_allowed",
    )
)

series_count = series_policy_df.count()
publication_count = publication_df.count()
if not series_count or publication_count != series_count:
    raise ValueError(
        "Profile and status must cover the same snapshot: "
        f"perfil={series_count}, status={publication_count}"
    )

joined_df = (
    series_policy_df.alias("policy")
    .join(
        publication_df.alias("publication"),
        on=["product_variant_id", "location_id"],
        how="inner",
    )
    .join(
        inventory_df.alias("inventory"),
        on=["product_variant_id", "location_id"],
        how="left",
    )
)
if joined_df.count() != series_count:
    raise AssertionError("The replenishment policy lost series during the join")

inventory_coverage = joined_df.filter(F.col("current_stock_qty").isNotNull()).count()
if inventory_coverage != series_count:
    raise ValueError(
        "Inventory must cover all series on the same operational date: "
        f"inventario={inventory_coverage}, series={series_count}, "
        f"data={inventory_reference_date}"
    )
print(f"Data do inventario: {inventory_reference_date}")
print(f"Forecast series: {series_count}")
print(f"Series with inventory: {inventory_coverage}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Manual replenishment decision
# MAGIC
# MAGIC Priority is based on stockout risk, projected inventory, and reorder point.
# MAGIC Purchase recommendations always require human approval.

# COMMAND ----------


def as_float(row, name: str) -> float | None:
    value = row[name]
    return float(value) if value is not None else None


created_at = datetime.now(timezone.utc).replace(tzinfo=None)
records = []
for row in joined_df.collect():
    decision = decide_replenishment(
        demand_segment=str(row["demand_segment"]),
        is_reorder_needed=bool(row["is_reorder_needed"]),
        is_reorder_needed_after_inbound=bool(row["is_reorder_needed_after_inbound"]),
        is_stockout_risk=bool(row["is_stockout_risk"]),
        suggested_purchase_qty=as_float(row, "suggested_purchase_qty"),
    )
    records.append(
        {
            "forecast_run_id": FORECAST_RUN_ID,
            "inventory_reference_date": inventory_reference_date,
            "product_variant_id": row["product_variant_id"],
            "location_id": row["location_id"],
            "demand_segment": row["demand_segment"],
            "confidence_level": row["confidence_level"],
            "forecast_publication_status": row["publication_status"],
            "forecast_review_required": bool(row["review_required"]),
            "forecast_automation_allowed": bool(row["automation_allowed"]),
            "current_stock_qty": as_float(row, "current_stock_qty"),
            "open_purchase_qty": as_float(row, "open_purchase_qty"),
            "lead_time_days": as_float(row, "lead_time_days"),
            "effective_reorder_point": as_float(row, "effective_reorder_point"),
            "stock_coverage_days": as_float(row, "stock_coverage_days"),
            "is_reorder_needed": bool(row["is_reorder_needed"]),
            "is_reorder_needed_after_inbound": bool(
                row["is_reorder_needed_after_inbound"]
            ),
            "is_stockout_risk": bool(row["is_stockout_risk"]),
            "suggested_purchase_qty": as_float(row, "suggested_purchase_qty"),
            "replenishment_strategy": decision.replenishment_strategy,
            "priority": decision.priority,
            "action_required": decision.action_required,
            "review_cadence": decision.review_cadence,
            "decision_reason": decision.decision_reason,
            "demand_basis": "inventory_trailing_90d",
            "forecast_zero_not_used": True,
            "manual_approval_required": True,
            "policy_version": POLICY_VERSION,
            "created_at": created_at,
        }
    )

replenishment_df = spark.createDataFrame(records)  # noqa: F821
if replenishment_df.count() != series_count:
    raise AssertionError("The replenishment policy does not cover every series")

show_result(
    replenishment_df.groupBy("demand_segment", "priority", "action_required")
    .count()
    .orderBy("demand_segment", "priority")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Idempotent, traceable persistence

# COMMAND ----------

if spark.catalog.tableExists(OUTPUT_TABLE):  # noqa: F821
    from delta.tables import DeltaTable

    target = DeltaTable.forName(spark, OUTPUT_TABLE)  # noqa: F821
    merge_condition = (
        "target.forecast_run_id = source.forecast_run_id "
        "AND target.inventory_reference_date = source.inventory_reference_date "
        "AND target.product_variant_id = source.product_variant_id "
        "AND target.location_id = source.location_id"
    )
    (
        target.alias("target")
        .merge(replenishment_df.alias("source"), merge_condition)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .whenNotMatchedBySourceDelete(
            condition=(
                f"target.forecast_run_id = '{FORECAST_RUN_ID}' "
                f"AND target.inventory_reference_date = DATE "
                f"'{inventory_reference_date.isoformat()}'"
            )
        )
        .execute()
    )
else:
    replenishment_df.write.format("delta").mode("append").saveAsTable(OUTPUT_TABLE)

persisted_df = spark.table(OUTPUT_TABLE).filter(  # noqa: F821
    (F.col("forecast_run_id") == F.lit(FORECAST_RUN_ID))
    & (F.col("inventory_reference_date") == F.lit(inventory_reference_date))
)
if persisted_df.count() != series_count:
    raise AssertionError("Replenishment policy persistence is incomplete")

show_result(
    persisted_df.groupBy("priority", "action_required").count().orderBy("priority")
)
print(f"Replenishment policy persisted to: {OUTPUT_TABLE}")
set_task_value("replenishment_policy_run_id", FORECAST_RUN_ID)
set_task_value(
    "replenishment_inventory_reference_date",
    inventory_reference_date.isoformat(),
)
print("REPLENISHMENT POLICY COMPLETE")
