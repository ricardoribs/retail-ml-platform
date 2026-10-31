# Databricks notebook source
# ruff: noqa: E402

# COMMAND ----------

"""Training and evaluation of regression models.

Trains Ridge, Random Forest, and Gradient-Boosted Trees with a temporal split,
compares results with MA3/MA6, and keeps fitted models in memory for inspection."""

from __future__ import annotations

import json
import sys
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

from ml.demand_forecasting.src.baseline import (
    evaluate_recursive_baselines,
)
from ml.demand_forecasting.src.config import (
    FORECAST_CONFIG,
    INTERMEDIATE_SCHEMA,
    MARTS_SCHEMA,
    fq_table,
    set_task_value,
)
from ml.demand_forecasting.src.modeling import (
    LABEL_COLUMN,
    PREDICTION_COLUMN,
    RECURSIVE_MODEL_VERSION,
    REGRESSION_MODEL_SPECS,
    SERIES_KEY_COLUMNS,
    build_temporal_split,
    collect_backtest_history,
    prepare_temporal_frames,
    select_best_model,
    shift_months,
    train_and_evaluate_models,
)
from ml.demand_forecasting.src.persistence import (
    METRICS_TABLE_NAME,
    MetricRunMetadata,
    persist_model_metrics,
    prepare_metric_records,
)

# COMMAND ----------


def show_result(df: DataFrame, rows: int = 100) -> None:
    """Display a DataFrame in Databricks, with a local execution fallback."""
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(rows, truncate=False)


FEATURE_MATRIX_TABLE = fq_table(INTERMEDIATE_SCHEMA, "int_forecast_feature_matrix")
MODEL_METRICS_TABLE = fq_table(MARTS_SCHEMA, METRICS_TABLE_NAME)
TEST_WINDOW_MONTHS = len(FORECAST_CONFIG.horizon_months)
TEMPORAL_SPLIT = build_temporal_split(
    FORECAST_CONFIG.cutoff_date,
    TEST_WINDOW_MONTHS,
)
MODEL_RUN_ID = f"model_{FORECAST_CONFIG.snapshot_id}"
MODEL_VERSION = RECURSIVE_MODEL_VERSION
BASELINE_FAMILIES = {
    "baseline_zero": "sanity_check",
    "baseline_ma3": "moving_average",
    "baseline_ma6": "moving_average",
    "baseline_seasonal": "seasonal_naive",
}
TARGET_MONTHS = tuple(
    shift_months(TEMPORAL_SPLIT.test_start, offset)
    for offset in range(TEST_WINDOW_MONTHS)
)

print("Training parameters")
print(f"  Feature table      : {FEATURE_MATRIX_TABLE}")
print(f"  Reference date     : {FORECAST_CONFIG.reference_date}")
print(f"  Cutoff date        : {FORECAST_CONFIG.cutoff_date}")
print(f"  Training end       : {TEMPORAL_SPLIT.train_end}")
print(f"  Test window        : {TEMPORAL_SPLIT.test_start} to {TEMPORAL_SPLIT.test_end}")
print(f"  Run ID             : {MODEL_RUN_ID}")
print(f"  Snapshot ID        : {FORECAST_CONFIG.snapshot_id}")
print(f"  Execution ID       : {FORECAST_CONFIG.execution_id}")
print(f"  Version            : {MODEL_VERSION}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Temporal split and eligibility
# MAGIC
# MAGIC Product-location eligibility is calculated using data only through the
# MAGIC start of the evaluation window. The three closed months through the cutoff
# MAGIC remain isolated for testing, so neither their target values nor sales affect
# MAGIC series selection or training.

# COMMAND ----------

feature_matrix_df = spark.table(FEATURE_MATRIX_TABLE)  # noqa: F821

fingerprint_columns = [
    F.coalesce(F.col(column).cast("string"), F.lit("<null>"))
    for column in sorted(feature_matrix_df.columns)
]
fingerprint_row = feature_matrix_df.agg(
    F.count(F.lit(1)).alias("row_count"),
    F.sum(F.pmod(F.xxhash64(*fingerprint_columns), F.lit(1_000_000_007))).alias(
        "content_hash"
    ),
).first()
INPUT_FINGERPRINT = (
    f"{fingerprint_row['row_count']}:{fingerprint_row['content_hash'] or 0}"
)
print(f"  Input fingerprint  : {INPUT_FINGERPRINT}")

expected_model_names = {
    *BASELINE_FAMILIES,
    *(spec.model_name for spec in REGRESSION_MODEL_SPECS),
}
existing_metric_rows = []
if spark.catalog.tableExists(MODEL_METRICS_TABLE):  # noqa: F821
    existing_metric_rows = (
        spark.table(MODEL_METRICS_TABLE)  # noqa: F821
        .filter(F.col("run_id") == F.lit(MODEL_RUN_ID))
        .select(
            "model_name",
            "model_version",
            F.get_json_object("parameters_json", "$.input_fingerprint").alias(
                "input_fingerprint"
            ),
        )
        .collect()
    )

existing_model_names = {row["model_name"] for row in existing_metric_rows}
existing_model_versions = {row["model_version"] for row in existing_metric_rows}
existing_fingerprints = {
    row["input_fingerprint"] for row in existing_metric_rows if row["input_fingerprint"]
}
can_reuse_metrics = (
    existing_model_names == expected_model_names
    and existing_model_versions == {MODEL_VERSION}
    and existing_fingerprints == {INPUT_FINGERPRINT}
)

if can_reuse_metrics:
    set_task_value("model_run_id", MODEL_RUN_ID)
    set_task_value("snapshot_id", FORECAST_CONFIG.snapshot_id)
    set_task_value("input_fingerprint", INPUT_FINGERPRINT)
    set_task_value("model_refreshed", False)
    message = (
        f"Reusing metrics for {FORECAST_CONFIG.snapshot_id}: "
        "the period and input data have not changed."
    )
    print(message)
    try:
        dbutils.notebook.exit(message)  # noqa: F821
    except NameError:
        pass

eligible_series_df, train_df, test_df = prepare_temporal_frames(
    feature_matrix_df=feature_matrix_df,
    split=TEMPORAL_SPLIT,
    min_history_months=FORECAST_CONFIG.min_history_months,
    min_positive_demand_months=FORECAST_CONFIG.min_positive_demand_months,
)

eligible_series_count = eligible_series_df.count()
training_rows = train_df.count()
test_rows = test_df.count()
history_states = collect_backtest_history(
    feature_matrix_df, test_df, TEMPORAL_SPLIT.train_end
)

if eligible_series_count == 0 or training_rows == 0 or test_rows == 0:
    raise ValueError(
        "Temporal split has no eligible data: "
        f"series={eligible_series_count}, train={training_rows}, test={test_rows}"
    )

train_limits = train_df.agg(
    F.min("demand_month").alias("first_month"),
    F.max("demand_month").alias("last_month"),
).first()
test_limits = test_df.agg(
    F.min("demand_month").alias("first_month"),
    F.max("demand_month").alias("last_month"),
).first()

if train_limits["last_month"] > TEMPORAL_SPLIT.train_end:
    raise AssertionError("Training contains an observation after the temporal boundary")
if test_limits["first_month"] < TEMPORAL_SPLIT.test_start:
    raise AssertionError("Test data contains an observation before the test window")

print(f"Eligible series at the end of training: {eligible_series_count}")
print(f"Training rows: {training_rows}")
print(f"Test rows: {test_rows}")

split_summary_df = spark.createDataFrame(  # noqa: F821
    [
        {
            "dataset": "train",
            "first_month": train_limits["first_month"],
            "last_month": train_limits["last_month"],
            "rows": training_rows,
        },
        {
            "dataset": "test",
            "first_month": test_limits["first_month"],
            "last_month": test_limits["last_month"],
            "rows": test_rows,
        },
    ]
)
show_result(split_summary_df.orderBy("dataset"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Baselines on the same test set
# MAGIC
# MAGIC Zero, MA3, MA6, and the same month from the previous year are calculated
# MAGIC over the operational horizon. Test actuals are used only to measure error.

# COMMAND ----------

baseline_rows = test_df.select(
    "demand_month", *SERIES_KEY_COLUMNS, LABEL_COLUMN
).collect()
baseline_metrics = evaluate_recursive_baselines(
    history_states, TARGET_MONTHS, baseline_rows
)
baseline_metric_rows = []
for model_name, metrics in baseline_metrics.items():
    baseline_metric_rows.append(
        {
            **metrics,
            "model_family": BASELINE_FAMILIES[model_name],
            "parameters_json": json.dumps(
                {
                    "evaluation_mode": "recursive_horizon",
                    "strategy": model_name,
                },
                sort_keys=True,
            ),
            "training_rows": 0,
            "training_seconds": 0.0,
        }
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Regression training
# MAGIC
# MAGIC Features: lags 1/2/3, moving averages 3/6, year, quarter, month, product,
# MAGIC category, location, location type, and state. Revenue and indicators observed
# MAGIC during the target month are excluded because they would cause forecast leakage.

# COMMAND ----------

trained_models, regression_prediction_frames, regression_metric_rows = (
    train_and_evaluate_models(
        train_df,
        test_df,
        history_states=history_states,
        target_months=TARGET_MONTHS,
    )
)

print(f"Trained models: {', '.join(trained_models)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Comparison and selection
# MAGIC
# MAGIC The primary criterion is the lowest WAPE, followed by MAE and RMSE. The
# MAGIC table includes `run_id`, version, parameters, and temporal boundaries for
# MAGIC persistence.

# COMMAND ----------

all_metric_rows = [*baseline_metric_rows, *regression_metric_rows]
for metric_row in all_metric_rows:
    parameters = json.loads(metric_row["parameters_json"])
    parameters.update(
        {
            "evaluation_mode": "recursive_horizon",
            "execution_id": FORECAST_CONFIG.execution_id,
            "input_fingerprint": INPUT_FINGERPRINT,
            "snapshot_id": FORECAST_CONFIG.snapshot_id,
        }
    )
    metric_row["parameters_json"] = json.dumps(parameters, sort_keys=True)

best_overall_metrics = select_best_model(all_metric_rows)
best_nonzero_metrics = select_best_model(
    [row for row in all_metric_rows if row["model_name"] != "baseline_zero"]
)
best_regression_metrics = select_best_model(regression_metric_rows)

metric_records = prepare_metric_records(
    metric_rows=all_metric_rows,
    metadata=MetricRunMetadata(
        run_id=MODEL_RUN_ID,
        model_version=MODEL_VERSION,
        train_start_month=train_limits["first_month"],
        train_end_month=TEMPORAL_SPLIT.train_end,
        test_start_month=TEMPORAL_SPLIT.test_start,
        test_end_month=TEMPORAL_SPLIT.test_end,
        cutoff_date=FORECAST_CONFIG.cutoff_date,
    ),
    best_model_name=str(best_overall_metrics["model_name"]),
)
metrics_df = persist_model_metrics(
    spark=spark,  # noqa: F821
    records=metric_records,
    table_name=MODEL_METRICS_TABLE,
)
show_result(metrics_df.orderBy("wape", "mae"))
print(f"Metrics persisted to: {MODEL_METRICS_TABLE}")

set_task_value("model_run_id", MODEL_RUN_ID)
set_task_value("snapshot_id", FORECAST_CONFIG.snapshot_id)
set_task_value("input_fingerprint", INPUT_FINGERPRINT)
set_task_value("model_refreshed", True)

print("Best overall model")
print(f"  model_name: {best_overall_metrics['model_name']}")
print(f"  WAPE: {best_overall_metrics['wape']}")
print(f"  MAE: {best_overall_metrics['mae']}")
print("Best non-zero model")
print(f"  model_name: {best_nonzero_metrics['model_name']}")
print(f"  WAPE: {best_nonzero_metrics['wape']}")
print(f"  MAE: {best_nonzero_metrics['mae']}")
print("Best regression model")
print(f"  model_name: {best_regression_metrics['model_name']}")
print(f"  WAPE: {best_regression_metrics['wape']}")
print(f"  MAE: {best_regression_metrics['mae']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Regression forecast evidence
# MAGIC
# MAGIC Negative forecasts are clipped to zero before metrics are calculated and
# MAGIC results are displayed.

# COMMAND ----------

prediction_frames = []
for model_name, predictions_df in regression_prediction_frames.items():
    prediction_frames.append(
        predictions_df.select(
            "demand_month",
            *SERIES_KEY_COLUMNS,
            F.col(LABEL_COLUMN).alias("actual_qty"),
            F.col(PREDICTION_COLUMN).alias("predicted_qty"),
        ).withColumn("model_name", F.lit(model_name))
    )

regression_predictions_df = prediction_frames[0]
for predictions_df in prediction_frames[1:]:
    regression_predictions_df = regression_predictions_df.unionByName(predictions_df)

negative_predictions = regression_predictions_df.filter(
    F.col("predicted_qty") < F.lit(0)
).count()
if negative_predictions != 0:
    raise AssertionError("Negative forecasts were generated")

print(f"Regression forecast rows: {regression_predictions_df.count()}")
print(f"Negative forecasts: {negative_predictions}")
show_result(
    regression_predictions_df.orderBy(
        "demand_month",
        "product_variant_id",
        "location_id",
        "model_name",
    )
)

# COMMAND ----------

print("TRAINING AND PERSISTENCE COMPLETE")
