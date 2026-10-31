# Databricks notebook source
# ruff: noqa: E402

"""Retrospective audit of inventory versus months without sales; read-only.

Reconstructs balances from the current snapshot and signed movements. This
reconstruction is a diagnostic hypothesis, not a training-ready feature: if the
movement ledger is incomplete, historical balances will be invalid."""

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


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Use the same two holdouts
# MAGIC
# MAGIC The goal is to assess whether months with zero sales could be months without
# MAGIC inventory. The current snapshot is used only for retrospective reconstruction.
# MAGIC **Do not use these balances for training or forecasting** without verifying
# MAGIC movement-ledger completeness and the availability of information at each cutoff.
# MAGIC This notebook does not write data or run as part of the job.

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
        raise ValueError(f"Holdout {cutoff} cohort differs from the eligible series")
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
# MAGIC ## 2. Reconstruct balances and test plausibility
# MAGIC
# MAGIC Inbound movements must be positive and outbound movements negative. The calculation works backward
# MAGIC backward from the `stg_stock_levels` balance to the tested months using movements
# MAGIC from each month. A negative balance makes the row **inconclusive**;
# MAGIC it is not counted as a confirmed stockout.

# COMMAND ----------

all_series = {observation["series"] for observation in observations}
stock_table = fq_table(STAGING_SCHEMA, "stg_stock_levels")
movement_table = fq_table(STAGING_SCHEMA, "stg_stock_movements")
snapshots = {}
for row in (
    spark.table(stock_table)  # noqa: F821
    .select(*SERIES_KEY_COLUMNS, "quantity_on_hand", "updated_at")
    .collect()
):
    key = (row["product_variant_id"], row["location_id"])
    if key in all_series:
        snapshots[key] = (row["quantity_on_hand"], row["updated_at"])

first_month = min(observation["month"] for observation in observations)
movement_rows = (
    spark.table(movement_table)  # noqa: F821
    .filter(F.col("occurred_at") >= F.lit(first_month))
    .select(*SERIES_KEY_COLUMNS, "movement_type", "quantity", "occurred_at")
    .collect()
)
movements = defaultdict(list)
bad_movement_series = set()
for row in (
    spark.table(movement_table)  # noqa: F821
    .filter(F.col("occurred_at").isNull())
    .select(*SERIES_KEY_COLUMNS)
    .collect()
):
    key = (row["product_variant_id"], row["location_id"])
    if key in all_series:
        bad_movement_series.add(key)
for row in movement_rows:
    key = (row["product_variant_id"], row["location_id"])
    if key not in all_series:
        continue
    movement_type = row["movement_type"]
    quantity = row["quantity"]
    if (
        quantity is None
        or row["occurred_at"] is None
        or (movement_type in {"purchase_in", "return_in"} and quantity < 0)
        or (movement_type in {"sale_out", "return_out"} and quantity > 0)
        or movement_type not in {"purchase_in", "return_in", "sale_out", "return_out"}
    ):
        bad_movement_series.add(key)
        continue
    movements[key].append(StockMovement(row["occurred_at"], float(quantity)))

print(
    f"Series audited: {len(all_series)}; snapshots: {len(snapshots)}; "
    f"movimentos lidos: {len(movement_rows)}; "
    f"series with invalid movements: {len(bad_movement_series)}"
)

status_counts = Counter()
summary_counts = defaultdict(Counter)
for observation in observations:
    cutoff = observation["cutoff"]
    month = observation["month"]
    series = observation["series"]
    location_type = observation["location_type"]
    actual_qty = observation["actual_qty"]
    snapshot = snapshots.get(series)
    if series in bad_movement_series:
        status = "invalid_movement_record"
    elif snapshot is None:
        status = "missing_snapshot"
    elif snapshot[0] is None or snapshot[1] is None or snapshot[0] < 0:
        status = "invalid_snapshot"
    else:
        inventory = reconstruct_monthly_inventory(
            snapshot_qty=float(snapshot[0]),
            snapshot_at=snapshot[1],
            movements=movements[series],
            month=month,
        )
        status = availability_status(inventory)

    sale_status = "sale" if actual_qty > 0 else "zero_sale"
    status_counts[(cutoff, month.isoformat(), location_type, sale_status, status)] += 1
    counters = summary_counts[(cutoff, location_type)]
    counters["rows"] += 1
    counters[sale_status] += 1
    counters[status] += 1
    if sale_status == "zero_sale" and status in {
        "no_positive_stock_all_month",
        "possible_stockout_within_month",
        "positive_stock_all_month",
    }:
        counters["valid_zero_sale"] += 1
        counters[f"zero_sale__{status}"] += 1
    if sale_status == "sale" and status == "no_positive_stock_all_month":
        counters["sale_without_positive_stock"] += 1

summary_rows = []
for (cutoff, location_type), counts in summary_counts.items():
    valid_zero = counts["valid_zero_sale"]
    no_stock_zero = counts["zero_sale__no_positive_stock_all_month"]
    summary_rows.append(
        {
            "cutoff": cutoff,
            "location_type": location_type,
            "rows": counts["rows"],
            "zero_sale_rows": counts["zero_sale"],
            "valid_zero_sale_rows": valid_zero,
            "zero_without_positive_stock": no_stock_zero,
            "zero_with_possible_stockout": counts[
                "zero_sale__possible_stockout_within_month"
            ],
            "zero_with_positive_stock": counts["zero_sale__positive_stock_all_month"],
            "no_stock_pct_of_valid_zeros": (
                round(100 * no_stock_zero / valid_zero, 2) if valid_zero else 0.0
            ),
            "negative_balance_rows": counts["invalid_negative_balance"],
            "missing_snapshot_rows": counts["missing_snapshot"],
            "old_snapshot_rows": counts["snapshot_before_month_end"],
            "invalid_source_rows": counts["invalid_snapshot"]
            + counts["invalid_movement_record"],
            "sale_without_positive_stock": counts["sale_without_positive_stock"],
        }
    )

status_rows = [
    {
        "cutoff": cutoff,
        "month": month,
        "location_type": location_type,
        "sale_status": sale_status,
        "inventory_status": status,
        "rows": count,
    }
    for (
        cutoff,
        month,
        location_type,
        sale_status,
        status,
    ), count in status_counts.items()
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Decision
# MAGIC
# MAGIC Interpret `zero_without_positive_stock` only when coverage is high and
# MAGIC `negative_balance_rows` and `sale_without_positive_stock` are low. The
# MAGIC absence of a positive balance **does not prove lost demand**; it only means
# MAGIC that the zero label may combine observed sales with possible unavailability.
# MAGIC If reconstruction fails or the fraction is small, do not change the target
# MAGIC or forecast. If the signal is material, validate the source balances before
# MAGIC building time-based features and rerunning both backtests.

# COMMAND ----------

show_result(
    spark.createDataFrame(summary_rows).orderBy(  # noqa: F821
        "cutoff", "location_type"
    )
)
show_result(
    spark.createDataFrame(status_rows).orderBy(  # noqa: F821
        "cutoff", "month", "location_type", "sale_status", "inventory_status"
    )
)

try:
    dbutils.notebook.exit(json.dumps({"summary": summary_rows}))  # noqa: F821
except NameError:
    pass
