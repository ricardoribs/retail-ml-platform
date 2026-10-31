# Databricks notebook source
# ruff: noqa: E402

# COMMAND ----------

"""Generates three horizons using the model selected for each product-location.

Each series receives the baseline that beat zero in its own recursive backtests.
Series without sufficient evidence remain visible in the review table, so the
applied zero-baseline forecast is not hidden."""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

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

from ml.demand_forecasting.src.config import (
    FORECAST_CONFIG,
    INTERMEDIATE_SCHEMA,
    MARTS_SCHEMA,
    fq_table,
    set_task_value,
    upstream_task_value,
)
from ml.demand_forecasting.src.modeling import (
    LABEL_COLUMN,
    SERIES_KEY_COLUMNS,
)
from ml.demand_forecasting.src.publication_policy import (
    STATUS_APPROVED_FORECAST,
    decide_validated_series_model,
)
from ml.demand_forecasting.src.segmentation import SeriesPolicy

# COMMAND ----------


def show_result(df: DataFrame, rows: int = 100) -> None:
    """Displays a DataFrame in Databricks, with a local fallback."""
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(rows, truncate=False)


FEATURE_MATRIX_TABLE = fq_table(INTERMEDIATE_SCHEMA, "int_forecast_feature_matrix")
FORECAST_TABLE = fq_table(MARTS_SCHEMA, "fct_demand_forecast")
POLICY_TABLE = fq_table(MARTS_SCHEMA, "fct_forecast_series_policy")
PUBLICATION_TABLE = fq_table(MARTS_SCHEMA, "fct_forecast_series_publication")
SERIES_SELECTION_TABLE = fq_table(MARTS_SCHEMA, "fct_forecast_series_model_selection")
FORECAST_RUN_ID = FORECAST_CONFIG.snapshot_id
MODEL_RUN_ID = upstream_task_value(
    "demand_forecast_train_evaluate",
    "model_run_id",
    f"model_{FORECAST_CONFIG.snapshot_id}",
)
EXPECTED_MODEL_RUN_ID = f"model_{FORECAST_CONFIG.snapshot_id}"
if MODEL_RUN_ID != EXPECTED_MODEL_RUN_ID:
    raise ValueError(
        f"Model run mismatch: expected={EXPECTED_MODEL_RUN_ID}, "
        f"received={MODEL_RUN_ID}"
    )
MODEL_REFRESHED = (
    upstream_task_value(
        "demand_forecast_train_evaluate", "model_refreshed", "true"
    ).lower()
    == "true"
)
POLICY_RUN_ID = upstream_task_value(
    "forecast_series_policy", "policy_run_id", FORECAST_CONFIG.snapshot_id
)
if POLICY_RUN_ID != FORECAST_CONFIG.snapshot_id:
    raise ValueError("The received policy does not match the execution cutoff")
SERIES_SELECTION_RUN_ID = upstream_task_value(
    "forecast_series_model_selection",
    "series_selection_run_id",
    f"series_selection_{FORECAST_CONFIG.snapshot_id}",
)
EXPECTED_SELECTION_RUN_ID = f"series_selection_{FORECAST_CONFIG.snapshot_id}"
if SERIES_SELECTION_RUN_ID != EXPECTED_SELECTION_RUN_ID:
    raise ValueError("The per-series selection does not match the execution snapshot")
FORECAST_MODEL_VERSION = "forecast_v5_series_champion_sparse_zero"
FORECAST_MONTHS = tuple(
    date.fromisoformat(f"{month}-01") for month in FORECAST_CONFIG.horizon_months
)

print("Final forecast parameters")
print(f"  Feature table: {FEATURE_MATRIX_TABLE}")
print(f"  Reference date: {FORECAST_CONFIG.reference_date}")
print(f"  Cutoff date: {FORECAST_CONFIG.cutoff_date}")
print(f"  Forecast months: {', '.join(FORECAST_CONFIG.horizon_months)}")
print(f"  Run ID: {FORECAST_RUN_ID}")
print(f"  Model run ID: {MODEL_RUN_ID}")
print(f"  Policy run ID: {POLICY_RUN_ID}")
print(f"  Series selection run ID: {SERIES_SELECTION_RUN_ID}")
print(f"  Execution ID: {FORECAST_CONFIG.execution_id}")
print(f"  Version: {FORECAST_MODEL_VERSION}")

expected_forecast_months = set(FORECAST_MONTHS)
can_reuse_forecast = False
if spark.catalog.tableExists(FORECAST_TABLE):  # noqa: F821
    persisted_snapshot_df = spark.table(FORECAST_TABLE).filter(  # noqa: F821
        F.col("run_id") == F.lit(FORECAST_RUN_ID)
    )
    persisted_snapshot_rows = (
        persisted_snapshot_df.select("forecast_month", "model_version")
        .distinct()
        .collect()
    )
    persisted_months = {row["forecast_month"] for row in persisted_snapshot_rows}
    persisted_versions = {row["model_version"] for row in persisted_snapshot_rows}
    incomplete_series = (
        persisted_snapshot_df.groupBy("product_variant_id", "location_id")
        .agg(
            F.count(F.lit(1)).alias("row_count"),
            F.countDistinct("horizon_month").alias("horizon_count"),
        )
        .filter(
            (F.col("row_count") != F.lit(len(FORECAST_MONTHS)))
            | (F.col("horizon_count") != F.lit(len(FORECAST_MONTHS)))
        )
        .limit(1)
        .count()
    )
    can_reuse_forecast = (
        persisted_months == expected_forecast_months
        and persisted_versions == {FORECAST_MODEL_VERSION}
        and incomplete_series == 0
    )

if not MODEL_REFRESHED and can_reuse_forecast:
    has_actionable_forecast = False
    if spark.catalog.tableExists(PUBLICATION_TABLE):  # noqa: F821
        persisted_publication_df = spark.table(PUBLICATION_TABLE).filter(  # noqa: F821
            F.col("run_id") == F.lit(FORECAST_RUN_ID)
        )
        has_publication_status = persisted_publication_df.limit(1).count() == 1
        has_unapproved_series = (
            persisted_publication_df.filter(
                F.col("publication_status") != F.lit(STATUS_APPROVED_FORECAST)
            )
            .limit(1)
            .count()
            == 1
        )
        has_actionable_forecast = has_publication_status and not has_unapproved_series

    set_task_value("snapshot_id", FORECAST_RUN_ID)
    set_task_value("model_run_id", MODEL_RUN_ID)
    set_task_value("forecast_refreshed", False)
    set_task_value(
        "has_actionable_forecast",
        str(has_actionable_forecast).lower(),
        required=True,
    )
    message = (
        f"Reusing forecast {FORECAST_RUN_ID}: "
        "the model, period, and input data have not changed."
    )
    print(message)
    try:
        dbutils.notebook.exit(message)  # noqa: F821
    except NameError:
        pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Series history and policy
# MAGIC
# MAGIC The as-of profile must cover exactly the series in the snapshot. The policy
# MAGIC was calculated before the forecast using only data available through the cutoff.

# COMMAND ----------

feature_matrix_df = spark.table(FEATURE_MATRIX_TABLE)  # noqa: F821
history_df = feature_matrix_df.filter(
    F.col("demand_month") <= F.lit(FORECAST_CONFIG.cutoff_date)
)

latest_window = Window.partitionBy(*SERIES_KEY_COLUMNS).orderBy(
    F.col("demand_month").desc()
)
latest_series_df = (
    history_df.withColumn("_row_number", F.row_number().over(latest_window))
    .filter(F.col("_row_number") == 1)
    .drop("_row_number")
)

recent_window = Window.partitionBy(*SERIES_KEY_COLUMNS).orderBy(
    F.col("demand_month").desc()
)
recent_history_df = (
    history_df.withColumn("_row_number", F.row_number().over(recent_window))
    .filter(F.col("_row_number") <= 12)
    .drop("_row_number")
)

latest_rows = latest_series_df.collect()
recent_rows = recent_history_df.collect()

if not spark.catalog.tableExists(POLICY_TABLE):  # noqa: F821
    raise ValueError("The per-series policy was not found for this forecast")
policy_rows = (
    spark.table(POLICY_TABLE)  # noqa: F821
    .filter(F.col("run_id") == F.lit(POLICY_RUN_ID))
    .select(
        "product_variant_id",
        "location_id",
        "demand_segment",
        "forecast_strategy",
        "confidence_level",
        "selection_reason",
        "history_month_count",
        "positive_demand_month_count",
        "trailing_positive_month_count",
        "months_since_last_sale",
    )
    .collect()
)
policy_by_key = {}
for row in policy_rows:
    key = (row["product_variant_id"], row["location_id"])
    if key in policy_by_key:
        raise ValueError("The policy contains duplicate product-location series")
    policy_by_key[key] = SeriesPolicy(
        demand_segment=str(row["demand_segment"]),
        forecast_strategy=str(row["forecast_strategy"]),
        confidence_level=str(row["confidence_level"]),
        selection_reason=str(row["selection_reason"]),
        history_month_count=int(row["history_month_count"]),
        positive_demand_month_count=int(row["positive_demand_month_count"]),
        trailing_positive_month_count=int(row["trailing_positive_month_count"]),
        months_since_last_sale=(
            int(row["months_since_last_sale"])
            if row["months_since_last_sale"] is not None
            else None
        ),
    )

forecast_keys = {
    tuple(row[column] for column in SERIES_KEY_COLUMNS) for row in latest_rows
}
if set(policy_by_key) != forecast_keys:
    raise ValueError(
        "The policy does not cover exactly the forecast series: "
        f"policy={len(policy_by_key)}, forecast={len(forecast_keys)}"
    )

if not latest_rows:
    raise ValueError("The feature matrix has no series through the cutoff date")

print(f"Series in forecast scope: {len(latest_rows)}")
print(f"Series covered by the policy: {len(policy_by_key)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Segmented publication evidence
# MAGIC
# MAGIC The global metric is still checked to ensure run integrity,
# MAGIC but the seasonal path is approved only by the segment's own recursive metric.

# COMMAND ----------

if not spark.catalog.tableExists(SERIES_SELECTION_TABLE):  # noqa: F821
    raise ValueError("The granular model selection was not found")
selection_rows = (
    spark.table(SERIES_SELECTION_TABLE)  # noqa: F821
    .filter(F.col("run_id") == F.lit(SERIES_SELECTION_RUN_ID))
    .select(
        "product_variant_id",
        "location_id",
        "selected_model_name",
        "validation_status",
        "selection_reason",
    )
    .collect()
)
selection_by_key = {}
for row in selection_rows:
    key = (row["product_variant_id"], row["location_id"])
    if key in selection_by_key:
        raise ValueError("The selection contains duplicate product-location series")
    selection_by_key[key] = row.asDict()
print(f"Series with granular backtests: {len(selection_by_key)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Recursive three-month forecast
# MAGIC
# MAGIC Each forecast month is added to its series history before building
# MAGIC the following month's features, without using future actual demand.

# COMMAND ----------

STATIC_COLUMNS = (
    "product_variant_id",
    "product_key",
    "sku",
    "product_name",
    "category_name",
    "location_id",
    "location_key",
    "location_name",
    "location_type",
    "city",
    "state",
)


def moving_average(history: list[tuple[date, float]], window: int) -> float:
    """Reproduces the moving averages used in the recursive backtest."""
    values = [qty for _, qty in history[-window:]]
    return max(sum(values) / len(values), 0.0) if values else 0.0


def seasonal_or_ma3(history: list[tuple[date, float]], target_month: date) -> float:
    """Apply the same recursive seasonal cascade used in the backtest."""
    previous_year_month = date(target_month.year - 1, target_month.month, 1)
    history_by_month = {month: qty for month, qty in history}
    if previous_year_month in history_by_month:
        return max(float(history_by_month[previous_year_month]), 0.0)
    return moving_average(history, 3)


series_states = {}
publication_records = []
created_at = datetime.now(timezone.utc).replace(tzinfo=None)
for latest in latest_rows:
    key = tuple(latest[column] for column in SERIES_KEY_COLUMNS)
    policy = policy_by_key[key]
    selection = selection_by_key.get(key)
    if selection is None:
        selection = {
            "selected_model_name": "baseline_zero",
            "validation_status": "insufficient_validation_signal",
            "selection_reason": (
                "Series lack coverage across all three historical windows; "
                "the zero baseline is retained for review."
            ),
        }
    decision = decide_validated_series_model(
        policy,
        selected_model_name=str(selection["selected_model_name"]),
        validation_status=str(selection["validation_status"]),
        selection_reason=str(selection["selection_reason"]),
    )
    series_states[key] = {
        "static": {column: latest[column] for column in STATIC_COLUMNS},
        "history": [],
        "policy": policy,
        "decision": decision,
    }
    publication_records.append(
        {
            "run_id": FORECAST_RUN_ID,
            "source_policy_run_id": POLICY_RUN_ID,
            "source_model_run_id": MODEL_RUN_ID,
            "product_variant_id": key[0],
            "location_id": key[1],
            "demand_segment": policy.demand_segment,
            "candidate_strategy": policy.forecast_strategy,
            "confidence_level": policy.confidence_level,
            "applied_model_name": decision.applied_model_name,
            "applied_strategy": decision.applied_strategy,
            "publication_status": decision.publication_status,
            "review_required": decision.review_required,
            "automation_allowed": decision.automation_allowed,
            "review_cadence": decision.review_cadence,
            "decision_reason": decision.decision_reason,
            "created_at": created_at,
        }
    )

for recent in recent_rows:
    key = tuple(recent[column] for column in SERIES_KEY_COLUMNS)
    series_states[key]["history"].append(
        (recent["demand_month"], float(recent[LABEL_COLUMN] or 0.0))
    )

for state in series_states.values():
    state["history"].sort(key=lambda item: item[0])

forecast_records = []
for horizon, target_month in enumerate(FORECAST_MONTHS, start=1):
    for state in series_states.values():
        decision = state["decision"]
        if decision.applied_model_name == "baseline_seasonal":
            forecast_qty = seasonal_or_ma3(state["history"], target_month)
        elif decision.applied_model_name == "baseline_ma3":
            forecast_qty = moving_average(state["history"], 3)
        elif decision.applied_model_name == "baseline_ma6":
            forecast_qty = moving_average(state["history"], 6)
        else:
            forecast_qty = 0.0

        state["history"].append((target_month, forecast_qty))
        forecast_records.append(
            {
                "run_id": FORECAST_RUN_ID,
                "forecast_month": target_month,
                "horizon_month": horizon,
                **state["static"],
                "forecast_demand_qty": float(max(forecast_qty, 0.0)),
                "model_name": decision.applied_model_name,
                "model_version": FORECAST_MODEL_VERSION,
                "created_at": created_at,
            }
        )

if not forecast_records:
    raise ValueError("No forecasts were generated")

forecast_df = spark.createDataFrame(forecast_records)  # noqa: F821
negative_count = forecast_df.filter(F.col("forecast_demand_qty") < F.lit(0.0)).count()
if negative_count:
    raise AssertionError("Negative forecasts were generated")

expected_rows = len(latest_rows) * len(FORECAST_MONTHS)
actual_rows = forecast_df.count()
if actual_rows != expected_rows:
    raise AssertionError(
        f"Unexpected forecast row count: expected={expected_rows}, "
        f"obtido={actual_rows}"
    )

print(f"Rows generated: {actual_rows}")
print(f"Negative forecasts: {negative_count}")
show_result(
    forecast_df.groupBy("horizon_month", "model_name")
    .count()
    .orderBy("horizon_month", "model_name")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Idempotent persistence

# COMMAND ----------

if spark.catalog.tableExists(FORECAST_TABLE):  # noqa: F821
    from delta.tables import DeltaTable

    target = DeltaTable.forName(spark, FORECAST_TABLE)  # noqa: F821
    merge_condition = (
        "target.run_id = source.run_id "
        "AND target.forecast_month = source.forecast_month "
        "AND target.product_variant_id = source.product_variant_id "
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

persisted_forecast_df = spark.table(FORECAST_TABLE).where(  # noqa: F821
    F.col("run_id") == F.lit(FORECAST_RUN_ID)
)
show_result(
    persisted_forecast_df.select(
        "run_id",
        "forecast_month",
        "horizon_month",
        "product_variant_id",
        "location_id",
        "forecast_demand_qty",
        "model_name",
        "model_version",
    ).orderBy("forecast_month", "product_variant_id", "location_id")
)
print(f"Forecast persisted to: {FORECAST_TABLE}")
print(f"Run ID: {FORECAST_RUN_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Publication status and review policy
# MAGIC
# MAGIC The status table keeps the applied strategy separate from the numeric
# MAGIC forecast table. Consumers should block automation when
# MAGIC `automation_allowed` is false and expose the defined review cadence.

# COMMAND ----------

publication_df = spark.createDataFrame(publication_records)  # noqa: F821
if publication_df.count() != len(latest_rows):
    raise AssertionError("The publication status does not cover every series")

if spark.catalog.tableExists(PUBLICATION_TABLE):  # noqa: F821
    from delta.tables import DeltaTable

    target = DeltaTable.forName(spark, PUBLICATION_TABLE)  # noqa: F821
    merge_condition = (
        "target.run_id = source.run_id "
        "AND target.product_variant_id = source.product_variant_id "
        "AND target.location_id = source.location_id"
    )
    (
        target.alias("target")
        .merge(publication_df.alias("source"), merge_condition)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .whenNotMatchedBySourceDelete(condition=f"target.run_id = '{FORECAST_RUN_ID}'")
        .execute()
    )
else:
    publication_df.write.format("delta").mode("append").saveAsTable(PUBLICATION_TABLE)

persisted_publication_df = spark.table(PUBLICATION_TABLE).filter(  # noqa: F821
    F.col("run_id") == F.lit(FORECAST_RUN_ID)
)
if persisted_publication_df.count() != len(publication_records):
    raise AssertionError("Publication status persistence is incomplete")

show_result(
    persisted_publication_df.groupBy(
        "demand_segment", "applied_model_name", "publication_status", "review_required"
    )
    .count()
    .orderBy("demand_segment", "applied_model_name")
)
print(f"Publication status persisted to: {PUBLICATION_TABLE}")

has_actionable_forecast = all(
    record["publication_status"] == STATUS_APPROVED_FORECAST
    for record in publication_records
)
set_task_value("snapshot_id", FORECAST_RUN_ID)
set_task_value("model_run_id", MODEL_RUN_ID)
set_task_value("forecast_refreshed", True)
set_task_value(
    "has_actionable_forecast",
    str(has_actionable_forecast).lower(),
    required=True,
)
print(f"Forecast is actionable for rankings: {has_actionable_forecast}")
print("FORECAST GENERATION COMPLETED")
