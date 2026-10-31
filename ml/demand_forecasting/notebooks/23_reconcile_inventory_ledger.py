# Databricks notebook source
# ruff: noqa: E402

"""Audits an inventory ledger reconciled using Bronze events only.

Inventory movements are the primary source. Paid sales and receipts supplement the
ledger only when no movement exists for the same reference, product, and location,
so duplicate events are not summed. Returns remain in the movements: return events
do not reliably identify the product/location for every exchange shipment.

This notebook is diagnostic, read-only, and outside the forecast job."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

if TYPE_CHECKING:
    dbutils: Any
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

from ml.demand_forecasting.src.availability import (
    StockMovement,
    availability_status,
    reconstruct_monthly_inventory,
)
from ml.demand_forecasting.src.config import (
    FORECAST_CONFIG,
    INTERMEDIATE_SCHEMA,
    STAGING_SCHEMA,
    fq_table,
)
from ml.demand_forecasting.src.modeling import (
    LABEL_COLUMN,
    SERIES_KEY_COLUMNS,
    build_temporal_split,
    prepare_temporal_frames,
    shift_months,
)


def show_result(df: DataFrame) -> None:
    """Display small tables without persisting results."""
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(100, truncate=False)


def inventory_status_for(
    *,
    series: tuple[int, int],
    month: date,
    snapshots: dict[tuple[int, int], tuple[float, object]],
    invalid_movement_series: set[tuple[int, int]],
    movements: dict[tuple[int, int], list[StockMovement]],
) -> str:
    """Classify availability without turning the audit into a feature."""
    if series in invalid_movement_series:
        return "invalid_movement_record"
    snapshot = snapshots.get(series)
    if snapshot is None:
        return "missing_snapshot"
    snapshot_qty, snapshot_at = snapshot
    if snapshot_qty is None or snapshot_at is None or snapshot_qty < 0:
        return "invalid_snapshot"
    inventory = reconstruct_monthly_inventory(
        snapshot_qty=float(snapshot_qty),
        snapshot_at=snapshot_at,
        movements=movements[series],
        month=month,
    )
    return availability_status(inventory)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Cutoffs and cohort matching the SKU-store backtests
# MAGIC
# MAGIC Bronze CSVs remain the only source. Separate cutoffs prevent future actuals
# MAGIC from entering the forecast; they are used here only to evaluate inventory
# MAGIC ledger quality retrospectively.

# COMMAND ----------

dbutils.widgets.text("cutoff_dates", "2025-11-30,2026-08-31")  # noqa: F821
cutoff_dates = tuple(
    date.fromisoformat(value.strip())
    for value in dbutils.widgets.get("cutoff_dates").split(",")  # noqa: F821
    if value.strip()
)
if not cutoff_dates or len(cutoff_dates) != len(set(cutoff_dates)):
    raise ValueError("Provide distinct cutoff dates in YYYY-MM-DD format")
for cutoff in cutoff_dates:
    if cutoff != shift_months(cutoff, 1) - timedelta(days=1):
        raise ValueError(f"The cutoff must be the last day of the month: {cutoff}")
    if cutoff > FORECAST_CONFIG.cutoff_date:
        raise ValueError(f"Cutoff is after the last closed month: {cutoff}")

feature_table = fq_table(INTERMEDIATE_SCHEMA, "int_forecast_feature_matrix")
feature_df = spark.table(feature_table).filter(  # noqa: F821
    F.col("demand_month") <= F.lit(max(cutoff_dates))
)
observations: list[dict[str, object]] = []
for cutoff in cutoff_dates:
    split = build_temporal_split(cutoff, test_months=3)
    eligible_df, _, test_df = prepare_temporal_frames(
        feature_matrix_df=feature_df,
        split=split,
        min_history_months=FORECAST_CONFIG.min_history_months,
        min_positive_demand_months=FORECAST_CONFIG.min_positive_demand_months,
    )
    rows = test_df.select(
        "demand_month", *SERIES_KEY_COLUMNS, "location_type", LABEL_COLUMN
    ).collect()
    series = {(row["product_variant_id"], row["location_id"]) for row in rows}
    expected = {
        (shift_months(split.test_start, offset), product_id, location_id)
        for offset in range(3)
        for product_id, location_id in series
    }
    observed = {
        (row["demand_month"], row["product_variant_id"], row["location_id"])
        for row in rows
    }
    if len(rows) != len(observed) or observed != expected:
        raise ValueError(f"Holdout {cutoff} contains duplicate or incomplete keys")
    if len(series) != eligible_df.count():
        raise ValueError(f"Holdout cohort {cutoff} differs from the eligible series")
    observations.extend(
        {
            "cutoff": cutoff.isoformat(),
            "month": row["demand_month"],
            "series": (row["product_variant_id"], row["location_id"]),
            "location_type": row["location_type"] or "__missing__",
            "actual_qty": float(row[LABEL_COLUMN]),
        }
        for row in rows
    )
    print(f"Cutoff {cutoff}: {len(series)} series and {len(rows)} series-months")


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build the reconciled ledger without duplicating events
# MAGIC
# MAGIC - `stg_stock_movements` is the primary signed source.
# MAGIC - An order is added as a supplemental outbound event only when it is `paid`
# MAGIC   and no `orders` movement exists for the same reference, SKU, and location.
# MAGIC - A receipt is added as a supplemental inbound event only when no equivalent
# MAGIC   `goods_receipts` movement exists.
# MAGIC - Returns remain only in the movement data because an exchange may remove a
# MAGIC   different SKU, and the return table does not provide a complete key for every outbound event.

# COMMAND ----------

all_series = {observation["series"] for observation in observations}
cohort_df = spark.createDataFrame(  # noqa: F821
    sorted(all_series), SERIES_KEY_COLUMNS
)
first_month = min(observation["month"] for observation in observations)
movement_table = fq_table(STAGING_SCHEMA, "stg_stock_movements")

movement_source_df = (
    spark.table(movement_table)  # noqa: F821
    .filter(F.col("occurred_at") >= F.lit(first_month))
    .join(F.broadcast(cohort_df), on=list(SERIES_KEY_COLUMNS), how="inner")
    .select(
        *SERIES_KEY_COLUMNS,
        "movement_type",
        "quantity",
        "reference_table",
        "reference_id",
        "occurred_at",
    )
)
valid_movement = (
    F.col("quantity").isNotNull()
    & F.col("occurred_at").isNotNull()
    & F.col("movement_type").isin("purchase_in", "return_in", "sale_out", "return_out")
    & ~(
        F.col("movement_type").isin("purchase_in", "return_in")
        & (F.col("quantity") < F.lit(0))
    )
    & ~(
        F.col("movement_type").isin("sale_out", "return_out")
        & (F.col("quantity") > F.lit(0))
    )
)
valid_movement_df = movement_source_df.filter(valid_movement)
invalid_movement_series = {
    (row["product_variant_id"], row["location_id"])
    for row in movement_source_df.filter(~valid_movement)
    .select(*SERIES_KEY_COLUMNS)
    .distinct()
    .collect()
}

order_reference_df = (
    valid_movement_df.filter(F.col("reference_table") == F.lit("orders"))
    .groupBy("reference_id", *SERIES_KEY_COLUMNS)
    .agg(F.sum(-F.col("quantity")).alias("movement_qty"))
)
paid_order_events_df = (
    spark.table(fq_table(STAGING_SCHEMA, "stg_order_items"))  # noqa: F821
    .join(
        spark.table(fq_table(STAGING_SCHEMA, "stg_orders")),  # noqa: F821
        on="order_id",
        how="inner",
    )
    .filter(
        (F.col("order_status") == F.lit("paid"))
        & (F.col("placed_at") >= F.lit(first_month))
    )
    .select(
        F.col("order_id").alias("reference_id"),
        "product_variant_id",
        "location_id",
        F.col("placed_at").alias("occurred_at"),
        "quantity_qty",
    )
    .join(F.broadcast(cohort_df), on=list(SERIES_KEY_COLUMNS), how="inner")
    .groupBy("reference_id", *SERIES_KEY_COLUMNS, "occurred_at")
    .agg(F.sum("quantity_qty").alias("event_qty"))
)
paid_order_matches_df = paid_order_events_df.join(
    order_reference_df,
    on=["reference_id", *SERIES_KEY_COLUMNS],
    how="inner",
)
paid_order_supplement_df = paid_order_events_df.join(
    order_reference_df.select("reference_id", *SERIES_KEY_COLUMNS),
    on=["reference_id", *SERIES_KEY_COLUMNS],
    how="left_anti",
).select(
    *SERIES_KEY_COLUMNS,
    F.lit("sale_out").alias("movement_type"),
    (-F.col("event_qty")).alias("quantity"),
    F.lit("orders").alias("reference_table"),
    "reference_id",
    "occurred_at",
    F.lit("paid_order_without_movement").alias("event_source"),
)

receipt_reference_df = (
    valid_movement_df.filter(F.col("reference_table") == F.lit("goods_receipts"))
    .groupBy("reference_id", *SERIES_KEY_COLUMNS)
    .agg(F.sum("quantity").alias("movement_qty"))
)
receipt_events_df = (
    spark.table(fq_table(STAGING_SCHEMA, "stg_goods_receipts"))  # noqa: F821
    .join(
        spark.table(  # noqa: F821
            fq_table(STAGING_SCHEMA, "stg_goods_receipt_items")
        ),
        on="goods_receipt_id",
        how="inner",
    )
    .join(
        spark.table(  # noqa: F821
            fq_table(STAGING_SCHEMA, "stg_purchase_order_items")
        ),
        on="purchase_order_item_id",
        how="inner",
    )
    .join(
        spark.table(fq_table(STAGING_SCHEMA, "stg_purchase_orders")),  # noqa: F821
        on="purchase_order_id",
        how="inner",
    )
    .filter(F.col("received_at") >= F.lit(first_month))
    .select(
        F.col("goods_receipt_id").alias("reference_id"),
        "product_variant_id",
        F.col("destination_location_id").alias("location_id"),
        "received_at",
        "quantity_received",
    )
    .join(F.broadcast(cohort_df), on=list(SERIES_KEY_COLUMNS), how="inner")
    .groupBy("reference_id", *SERIES_KEY_COLUMNS, "received_at")
    .agg(F.sum("quantity_received").alias("event_qty"))
    .withColumnRenamed("received_at", "occurred_at")
)
receipt_matches_df = receipt_events_df.join(
    receipt_reference_df,
    on=["reference_id", *SERIES_KEY_COLUMNS],
    how="inner",
)
receipt_supplement_df = receipt_events_df.join(
    receipt_reference_df.select("reference_id", *SERIES_KEY_COLUMNS),
    on=["reference_id", *SERIES_KEY_COLUMNS],
    how="left_anti",
).select(
    *SERIES_KEY_COLUMNS,
    F.lit("purchase_in").alias("movement_type"),
    F.col("event_qty").alias("quantity"),
    F.lit("goods_receipts").alias("reference_table"),
    "reference_id",
    "occurred_at",
    F.lit("goods_receipt_without_movement").alias("event_source"),
)

base_ledger_df = valid_movement_df.select(
    *SERIES_KEY_COLUMNS,
    "movement_type",
    "quantity",
    "reference_table",
    "reference_id",
    "occurred_at",
    F.lit("stock_movement").alias("event_source"),
)
reconciled_ledger_df = base_ledger_df.unionByName(paid_order_supplement_df).unionByName(
    receipt_supplement_df
)

reconciliation_rows = [
    {
        "source": "stock_movement",
        "event_groups": base_ledger_df.count(),
        "supplemented_groups": 0,
        "quantity_mismatch_groups": 0,
    },
    {
        "source": "paid_order",
        "event_groups": paid_order_events_df.count(),
        "supplemented_groups": paid_order_supplement_df.count(),
        "quantity_mismatch_groups": paid_order_matches_df.filter(
            F.abs(F.col("event_qty") - F.col("movement_qty")) > F.lit(0.001)
        ).count(),
    },
    {
        "source": "goods_receipt",
        "event_groups": receipt_events_df.count(),
        "supplemented_groups": receipt_supplement_df.count(),
        "quantity_mismatch_groups": receipt_matches_df.filter(
            F.abs(F.col("event_qty") - F.col("movement_qty")) > F.lit(0.001)
        ).count(),
    },
    {
        "source": "returns",
        "event_groups": valid_movement_df.filter(
            F.col("reference_table") == F.lit("returns")
        ).count(),
        "supplemented_groups": 0,
        "quantity_mismatch_groups": 0,
    },
]
print(json.dumps({"reconciliation": reconciliation_rows}, ensure_ascii=False))


# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Compare the original and reconciled reconstructions
# MAGIC
# MAGIC A supplemental event is acceptable only if its reference key is absent from
# MAGIC the ledger. Therefore, `status_changed_from_movement_only` measures the
# MAGIC verifiable effect of including other Bronze events without inventing balances.

# COMMAND ----------

stock_table = fq_table(STAGING_SCHEMA, "stg_stock_levels")
snapshots = {
    (row["product_variant_id"], row["location_id"]): (
        row["quantity_on_hand"],
        row["updated_at"],
    )
    for row in spark.table(stock_table)  # noqa: F821
    .join(F.broadcast(cohort_df), on=list(SERIES_KEY_COLUMNS), how="inner")
    .select(*SERIES_KEY_COLUMNS, "quantity_on_hand", "updated_at")
    .collect()
}


def collect_movements(df: DataFrame) -> dict[tuple[int, int], list[StockMovement]]:
    """Convert the Spark ledger for the pure reconstruction function."""
    grouped: dict[tuple[int, int], list[StockMovement]] = defaultdict(list)
    for row in df.select(*SERIES_KEY_COLUMNS, "quantity", "occurred_at").collect():
        key = (row["product_variant_id"], row["location_id"])
        grouped[key].append(StockMovement(row["occurred_at"], float(row["quantity"])))
    return grouped


movement_only = collect_movements(base_ledger_df)
reconciled = collect_movements(reconciled_ledger_df)
summary_counts = defaultdict(Counter)
for observation in observations:
    cutoff = observation["cutoff"]
    month = observation["month"]
    series = observation["series"]
    location_type = observation["location_type"]
    actual_qty = observation["actual_qty"]
    base_status = inventory_status_for(
        series=series,
        month=month,
        snapshots=snapshots,
        invalid_movement_series=invalid_movement_series,
        movements=movement_only,
    )
    reconciled_status = inventory_status_for(
        series=series,
        month=month,
        snapshots=snapshots,
        invalid_movement_series=invalid_movement_series,
        movements=reconciled,
    )
    counters = summary_counts[(cutoff, location_type)]
    counters["rows"] += 1
    counters["zero_sale_rows" if actual_qty == 0 else "sale_rows"] += 1
    counters[reconciled_status] += 1
    if base_status != reconciled_status:
        counters["status_changed_from_movement_only"] += 1
    if actual_qty == 0 and reconciled_status in {
        "no_positive_stock_all_month",
        "possible_stockout_within_month",
        "positive_stock_all_month",
    }:
        counters["valid_zero_sale_rows"] += 1
        counters[f"zero_sale__{reconciled_status}"] += 1

summary_rows = []
for (cutoff, location_type), counts in summary_counts.items():
    valid_zero = counts["valid_zero_sale_rows"]
    zero_without_stock = counts["zero_sale__no_positive_stock_all_month"]
    summary_rows.append(
        {
            "cutoff": cutoff,
            "location_type": location_type,
            "rows": counts["rows"],
            "zero_sale_rows": counts["zero_sale_rows"],
            "valid_zero_sale_rows": valid_zero,
            "zero_without_positive_stock": zero_without_stock,
            "zero_with_possible_stockout": counts[
                "zero_sale__possible_stockout_within_month"
            ],
            "zero_with_positive_stock": counts["zero_sale__positive_stock_all_month"],
            "no_stock_pct_of_valid_zeros": (
                round(100 * zero_without_stock / valid_zero, 2) if valid_zero else 0.0
            ),
            "negative_balance_rows": counts["invalid_negative_balance"],
            "old_snapshot_rows": counts["snapshot_before_month_end"],
            "invalid_source_rows": counts["invalid_snapshot"]
            + counts["invalid_movement_record"],
            "status_changed_from_movement_only": counts[
                "status_changed_from_movement_only"
            ],
        }
    )

show_result(
    spark.createDataFrame(reconciliation_rows).orderBy("source")  # noqa: F821
)
show_result(
    spark.createDataFrame(summary_rows).orderBy("cutoff", "location_type")  # noqa: F821
)

try:
    dbutils.notebook.exit(  # noqa: F821
        json.dumps(
            {
                "reconciliation": reconciliation_rows,
                "summary": summary_rows,
            }
        )
    )
except NameError:
    pass
