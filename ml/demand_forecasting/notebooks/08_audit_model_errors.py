# Databricks notebook source
# ruff: noqa: E402

"""Isolated audit of the selected model's error; does not write tables.

Reproduces the window and model recorded in `fct_forecast_model_metrics`, verifies
data identity, and calculates error by location and actual-demand band. Run manually;
this notebook is not part of the daily job."""

from __future__ import annotations

import json
import math
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

from ml.demand_forecasting.src.config import (
    FORECAST_CONFIG,
    INTERMEDIATE_SCHEMA,
    MARTS_SCHEMA,
    fq_table,
)
from ml.demand_forecasting.src.modeling import (
    LABEL_COLUMN,
    PREDICTION_COLUMN,
    RECURSIVE_MODEL_VERSION,
    REGRESSION_MODEL_SPECS,
    TemporalSplit,
    collect_backtest_history,
    prepare_temporal_frames,
    shift_months,
    train_and_evaluate_models,
)
from ml.demand_forecasting.src.persistence import METRICS_TABLE_NAME


def show_result(df: DataFrame) -> None:
    """Display the small number of groups produced by the audit."""
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(100, truncate=False)


def summarize_errors(
    audit_df: DataFrame,
    group_columns: tuple[str, ...],
    total_absolute_error: float,
) -> DataFrame:
    """Sum errors by segment; WAPE is null when actual demand is zero."""
    grouped_df = audit_df.groupBy(*group_columns).agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.when(F.col("actual_qty") == 0, 1).otherwise(0)).alias("zero_rows"),
        F.sum("actual_qty").alias("actual_qty"),
        F.sum("predicted_qty").alias("predicted_qty"),
        F.sum("absolute_error").alias("absolute_error"),
        F.sum(
            F.when(F.col("actual_qty") == 0, F.col("absolute_error")).otherwise(0.0)
        ).alias("zero_actual_absolute_error"),
    )
    return (
        grouped_df.withColumn(
            "zero_pct", F.round(100 * F.col("zero_rows") / F.col("rows"), 2)
        )
        .withColumn("mae", F.round(F.col("absolute_error") / F.col("rows"), 2))
        .withColumn(
            "wape_pct",
            F.when(
                F.col("actual_qty") > 0,
                F.round(100 * F.col("absolute_error") / F.col("actual_qty"), 2),
            ),
        )
        .withColumn(
            "error_share_pct",
            F.when(
                F.lit(total_absolute_error) > 0,
                F.round(100 * F.col("absolute_error") / F.lit(total_absolute_error), 2),
            ),
        )
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Pin the run and verify that the data has not changed
# MAGIC
# MAGIC The comparison stops if the fingerprint, parameters, or
# MAGIC the test size differ from the published run. This notebook does not write
# MAGIC to any table.
# MAGIC criada ou atualizada por este notebook.

# COMMAND ----------

dbutils.widgets.text("model_run_id", "model_forecast_2026_08")  # noqa: F821
model_run_id = dbutils.widgets.get("model_run_id").strip()  # noqa: F821
if not model_run_id:
    raise ValueError("Informe model_run_id, por exemplo model_forecast_2026_08")

feature_table = fq_table(INTERMEDIATE_SCHEMA, "int_forecast_feature_matrix")
metrics_table = fq_table(MARTS_SCHEMA, METRICS_TABLE_NAME)
reference_rows = (
    spark.table(metrics_table)  # noqa: F821
    .filter(
        (F.col("run_id") == model_run_id)
        & F.col("model_family").isin(*(spec.family for spec in REGRESSION_MODEL_SPECS))
    )
    .orderBy("wape", "mae", "rmse", "model_name")
    .limit(1)
    .collect()
)
if len(reference_rows) != 1:
    raise ValueError(
        f"Expected one best-regression row for {model_run_id}; "
        f"encontradas {len(reference_rows)}"
    )
reference = reference_rows[0]
spec = next(
    (
        item
        for item in REGRESSION_MODEL_SPECS
        if item.model_name == reference["model_name"]
    ),
    None,
)
if spec is None:
    raise ValueError(
        "The registered best model is not a regression model available for this audit: "
        f"{reference['model_name']}"
    )
if reference["model_version"] not in {"regression_v1", RECURSIVE_MODEL_VERSION}:
    raise ValueError(f"Unsupported model version: {reference['model_version']}")

parameters = json.loads(reference["parameters_json"])
if any(parameters.get(key) != value for key, value in spec.parameters.items()):
    raise ValueError("Published hyperparameters differ from the current code")
expected_fingerprint = parameters.get("input_fingerprint")
if not expected_fingerprint:
    raise ValueError("Run has no input_fingerprint; comparing retraining is unsafe")

feature_matrix_df = spark.table(feature_table)  # noqa: F821
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
current_fingerprint = (
    f"{fingerprint_row['row_count']}:{fingerprint_row['content_hash'] or 0}"
)
if current_fingerprint != expected_fingerprint:
    raise ValueError(
        "A matriz de features mudou desde o run publicado: "
        f"expected={expected_fingerprint}, current={current_fingerprint}"
    )

split = TemporalSplit(
    train_end=reference["train_end_month"],
    test_start=reference["test_start_month"],
    test_end=reference["test_end_month"],
)
_, train_df, test_df = prepare_temporal_frames(
    feature_matrix_df=feature_matrix_df,
    split=split,
    min_history_months=FORECAST_CONFIG.min_history_months,
    min_positive_demand_months=FORECAST_CONFIG.min_positive_demand_months,
)
training_rows = train_df.count()
test_rows = test_df.count()
if training_rows != reference["training_rows"]:
    raise ValueError(
        f"Training has {training_rows} rows; the published run has "
        f"{reference['training_rows']}"
    )
if test_rows != reference["evaluated_rows"]:
    raise ValueError(
        f"Test has {test_rows} rows; the published run has {reference['evaluated_rows']}"
    )

print(f"Run auditado: {model_run_id}")
print(f"Modelo: {spec.model_name} ({reference['model_version']})")
print(f"Janela de teste: {split.test_start} a {split.test_end}")
print(f"Fingerprint confirmado: {current_fingerprint}")
print(f"Rows: train={training_rows}, test={test_rows}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Reevaluate only the winning model
# MAGIC
# MAGIC Use the same Spark ML pipeline and non-negative post-processing as the
# MAGIC official training run. V1 reproduces the earlier one-step evaluation; V2
# MAGIC reproduces the three recursive steps. Forecasts remain in memory only.

# COMMAND ----------

if reference["model_version"] == RECURSIVE_MODEL_VERSION:
    history_states = collect_backtest_history(
        feature_matrix_df, test_df, split.train_end
    )
    test_month_count = (
        (split.test_end.year - split.test_start.year) * 12
        + split.test_end.month
        - split.test_start.month
        + 1
    )
    target_months = tuple(
        shift_months(split.test_start, offset) for offset in range(test_month_count)
    )
    _, prediction_frames, metric_rows = train_and_evaluate_models(
        train_df,
        test_df,
        specs=(spec,),
        history_states=history_states,
        target_months=target_months,
    )
else:
    _, prediction_frames, metric_rows = train_and_evaluate_models(
        train_df, test_df, specs=(spec,), legacy_one_step=True
    )
recomputed = metric_rows[0]
for metric_name in ("mae", "rmse", "mape", "wape"):
    published_value = reference[metric_name]
    computed_value = recomputed[metric_name]
    if published_value is None or computed_value is None:
        raise ValueError(f"Metric {metric_name} is missing from the run or retraining")
    print(
        f"{metric_name.upper()}: publicado={published_value:.4f}, "
        f"recalculado={computed_value:.4f}"
    )
    if not math.isclose(
        float(computed_value), float(published_value), rel_tol=0.01, abs_tol=0.01
    ):
        raise ValueError(
            f"Metric {metric_name} did not reproduce the published run; "
            "do not interpret the segments below"
        )

# COMMAND ----------

predictions_df = prediction_frames[spec.model_name]
audit_df = (
    predictions_df.select(
        "demand_month",
        "product_variant_id",
        "location_id",
        "location_name",
        "location_type",
        F.col(LABEL_COLUMN).alias("actual_qty"),
        F.col(PREDICTION_COLUMN).alias("predicted_qty"),
    )
    .withColumn("absolute_error", F.abs(F.col("actual_qty") - F.col("predicted_qty")))
    .withColumn(
        "demand_band",
        F.when(F.col("actual_qty") == 0, F.lit("0"))
        .when(F.col("actual_qty") <= 1, F.lit("(0,1]"))
        .when(F.col("actual_qty") <= 5, F.lit("(1,5]"))
        .otherwise(F.lit(">5")),
    )
)
if audit_df.count() != test_rows:
    raise AssertionError("The audit lost test forecast rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Where is the error concentrated?
# MAGIC
# MAGIC WAPE is null when actual demand is zero; for this range, track
# MAGIC `absolute_error` and `error_share_pct`. The ranges use actuals from the
# MAGIC the holdout only to diagnose errors, never for training.

# COMMAND ----------

total_absolute_error = audit_df.agg(F.sum("absolute_error")).first()[0]

print("Error by location (stores and warehouses)")
show_result(
    summarize_errors(
        audit_df,
        ("location_id", "location_name", "location_type"),
        total_absolute_error,
    ).orderBy("location_type", F.desc("wape_pct"))
)

print("Error by actual-demand range")
show_result(
    summarize_errors(audit_df, ("demand_band",), total_absolute_error).orderBy(
        F.when(F.col("demand_band") == "0", 0)
        .when(F.col("demand_band") == "(0,1]", 1)
        .when(F.col("demand_band") == "(1,5]", 2)
        .otherwise(3)
    )
)

print("Error by location and actual-demand range")
show_result(
    summarize_errors(
        audit_df,
        ("location_id", "location_name", "location_type", "demand_band"),
        total_absolute_error,
    ).orderBy("location_type", "location_name", "demand_band")
)
