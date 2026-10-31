# Databricks notebook source
# ruff: noqa: E402

"""Manual three-month backtest for series eligible for regression.

Use a historical cutoff to measure the forecast as it would have been produced at
that time. Includes an L1 candidate for intermittent demand, still outside the job.
No tables or models are persisted. Spark ML training may take a while."""

from __future__ import annotations

import json
import sys
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

from ml.demand_forecasting.src.baseline import evaluate_recursive_baselines
from ml.demand_forecasting.src.config import (
    FORECAST_CONFIG,
    INTERMEDIATE_SCHEMA,
    fq_table,
)
from ml.demand_forecasting.src.modeling import (
    INTERMITTENT_MODEL_SPECS,
    LABEL_COLUMN,
    PREDICTION_COLUMN,
    REGRESSION_MODEL_SPECS,
    SERIES_KEY_COLUMNS,
    build_temporal_split,
    collect_backtest_history,
    prepare_temporal_frames,
    shift_months,
    train_and_evaluate_models,
)
from ml.demand_forecasting.src.two_stage import (
    THRESHOLDS,
    evaluate_two_stage_candidates,
)


def show_result(df: DataFrame) -> None:
    """Displays compact results in Databricks or in the terminal."""
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(100, truncate=False)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Choose a historical period
# MAGIC
# MAGIC To evaluate seasonality from September through November 2025, use the
# MAGIC `2025-11-30` cutoff. The default cutoff evaluates the three most recent
# MAGIC closed months.
# MAGIC `gradient_boosted_trees_l1` minimizes absolute rather than squared error;
# MAGIC `two_stage_gbt` separates sales occurrence from positive quantity and first
# MAGIC evaluates `P(sale) * E[quantity | sale]`; the thresholds are
# MAGIC only decision alternatives. Both are experimental. `all` trains
# MAGIC searched across all five options and is more expensive. Start by selecting
# MAGIC one option directly.

# COMMAND ----------

dbutils.widgets.text("cutoff_date", FORECAST_CONFIG.cutoff_date.isoformat())  # noqa: F821
dbutils.widgets.dropdown(  # noqa: F821
    "model_name",
    "two_stage_gbt",
    [
        "two_stage_gbt",
        "gradient_boosted_trees_l1",
        "gradient_boosted_trees",
        "random_forest",
        "ridge_regression",
        "all",
    ],
)
dbutils.widgets.dropdown(  # noqa: F821
    "occurrence_threshold",
    "all",
    ["0.10", "0.25", "0.50", "0.75", "all"],
)
cutoff_date = date.fromisoformat(dbutils.widgets.get("cutoff_date"))  # noqa: F821
model_name = dbutils.widgets.get("model_name")  # noqa: F821
occurrence_threshold = dbutils.widgets.get("occurrence_threshold")  # noqa: F821
if cutoff_date != shift_months(cutoff_date, 1) - timedelta(days=1):
    raise ValueError("cutoff_date must be the last day of a closed month")
if cutoff_date > FORECAST_CONFIG.cutoff_date:
    raise ValueError("cutoff_date cannot be after the last closed month")

candidate_specs = (*REGRESSION_MODEL_SPECS, *INTERMITTENT_MODEL_SPECS)
specs = tuple(
    spec
    for spec in candidate_specs
    if model_name == "all" or spec.model_name == model_name
)
if not specs and model_name != "two_stage_gbt":
    raise ValueError(f"Modelo desconhecido: {model_name}")
if occurrence_threshold == "all":
    two_stage_thresholds = THRESHOLDS
else:
    two_stage_thresholds = (float(occurrence_threshold),)

split = build_temporal_split(cutoff_date, test_months=3)
target_months = tuple(shift_months(split.test_start, offset) for offset in range(3))
feature_matrix_df = spark.table(  # noqa: F821
    fq_table(INTERMEDIATE_SCHEMA, "int_forecast_feature_matrix")
)
_, train_df, test_df = prepare_temporal_frames(
    feature_matrix_df,
    split,
    FORECAST_CONFIG.min_history_months,
    FORECAST_CONFIG.min_positive_demand_months,
)
history_states = collect_backtest_history(feature_matrix_df, test_df, split.train_end)

print(f"Training through: {split.train_end}")
print(f"Teste recursivo: {target_months}")
print(f"Eligible series evaluated: {len(history_states)}")
print(f"Training rows: {train_df.count()}; test rows: {test_df.count()}")

for dataset_name, frame in (("train", train_df), ("test", test_df)):
    demand_profile = frame.agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.when(F.col(LABEL_COLUMN) == 0, 1).otherwise(0)).alias("zero_rows"),
        F.avg(LABEL_COLUMN).alias("mean_demand"),
        F.expr("percentile_approx(demand_qty, 0.5)").alias("p50_demand"),
        F.expr("percentile_approx(demand_qty, 0.9)").alias("p90_demand"),
        F.expr("percentile_approx(demand_qty, 0.99)").alias("p99_demand"),
    ).withColumn("dataset", F.lit(dataset_name))
    show_result(demand_profile)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Compare models and benchmarks
# MAGIC
# MAGIC Actual values for each test month are used only to measure error.
# MAGIC The zero benchmark is a sanity check, not a publishable forecast.

# COMMAND ----------

actual_rows = test_df.select(
    "demand_month", *SERIES_KEY_COLUMNS, LABEL_COLUMN
).collect()
baseline_metrics = evaluate_recursive_baselines(
    history_states, target_months, actual_rows
)
prediction_frames = {}
regression_metrics = []
if specs:
    _, prediction_frames, regression_metrics = train_and_evaluate_models(
        train_df,
        test_df,
        specs=specs,
        history_states=history_states,
        target_months=target_months,
    )
if model_name in {"two_stage_gbt", "all"}:
    two_stage_frames, two_stage_metrics = evaluate_two_stage_candidates(
        train_df,
        test_df,
        history_states,
        target_months,
        thresholds=two_stage_thresholds,
    )
    prediction_frames.update(two_stage_frames)
    regression_metrics.extend(two_stage_metrics)
metric_rows = [*baseline_metrics.values(), *regression_metrics]
metric_summary = [
    {
        "model_name": row["model_name"],
        "wape_pct": float(row["wape"]),
        "mae": float(row["mae"]),
        "rmse": float(row["rmse"]),
        "evaluated_rows": int(row["evaluated_rows"]),
    }
    for row in metric_rows
]
show_result(spark.createDataFrame(metric_summary).orderBy("wape_pct"))  # noqa: F821

# The compact output supports running this notebook manually through the Jobs API
# without losing the metrics displayed in the table. This notebook does not modify tables.
backtest_result = {
    "cutoff_date": cutoff_date.isoformat(),
    "occurrence_threshold": occurrence_threshold,
    "target_months": [month.isoformat() for month in target_months],
    "models": [
        {
            "model_name": row["model_name"],
            "wape_pct": round(float(row["wape"]), 6),
            "mae": round(float(row["mae"]), 6),
            "rmse": round(float(row["rmse"]), 6),
            "evaluated_rows": int(row["evaluated_rows"]),
            "parameters": json.loads(row.get("parameters_json", "{}")),
        }
        for row in metric_rows
    ],
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Check the business scale
# MAGIC
# MAGIC A correct total can hide errors that cancel out across products. Inspect
# MAGIC all three months, locations, and the forecast distribution before relying
# MAGIC on the ranking. Actual-demand bands are for diagnosis only.

# COMMAND ----------

for selected_name, predictions_df in prediction_frames.items():
    print(f"Modelo: {selected_name}")
    prediction_profile = predictions_df.agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.when(F.col(LABEL_COLUMN) == 0, 1).otherwise(0)).alias(
            "actual_zero_rows"
        ),
        F.sum(F.when(F.col(PREDICTION_COLUMN) <= 1e-9, 1).otherwise(0)).alias(
            "predicted_zero_rows"
        ),
        F.sum(
            F.when(
                (F.col(PREDICTION_COLUMN) > 1e-9) & (F.col(LABEL_COLUMN) > 0),
                1,
            ).otherwise(0)
        ).alias("correctly_predicted_sale_rows"),
        F.sum(
            F.when(
                (F.col(PREDICTION_COLUMN) > 1e-9) & (F.col(LABEL_COLUMN) == 0),
                1,
            ).otherwise(0)
        ).alias("false_sale_rows"),
        F.sum(
            F.when(
                (F.col(PREDICTION_COLUMN) <= 1e-9) & (F.col(LABEL_COLUMN) > 0),
                1,
            ).otherwise(0)
        ).alias("missed_sale_rows"),
        F.avg(PREDICTION_COLUMN).alias("predicted_mean"),
        F.stddev_pop(PREDICTION_COLUMN).alias("predicted_stddev"),
        F.min(PREDICTION_COLUMN).alias("predicted_min"),
        F.expr("percentile_approx(prediction, 0.5)").alias("predicted_p50"),
        F.expr("percentile_approx(prediction, 0.9)").alias("predicted_p90"),
        F.expr("percentile_approx(prediction, 0.99)").alias("predicted_p99"),
        F.max(PREDICTION_COLUMN).alias("predicted_max"),
    )
    show_result(prediction_profile)

    if "occurrence_probability" in predictions_df.columns:
        show_result(
            predictions_df.groupBy((F.col(LABEL_COLUMN) > 0).alias("had_sale"))
            .agg(
                F.count(F.lit(1)).alias("rows"),
                F.avg("occurrence_probability").alias("mean_sale_probability"),
                F.avg("positive_qty_prediction").alias("mean_positive_qty"),
                F.avg(PREDICTION_COLUMN).alias("mean_final_prediction"),
            )
            .orderBy("had_sale")
        )

    by_month = predictions_df.groupBy("demand_month").agg(
        F.sum(LABEL_COLUMN).alias("actual_qty"),
        F.sum(PREDICTION_COLUMN).alias("predicted_qty"),
        F.sum(F.abs(F.col(LABEL_COLUMN) - F.col(PREDICTION_COLUMN))).alias(
            "sku_location_absolute_error"
        ),
    )
    show_result(by_month.orderBy("demand_month"))

    by_location = predictions_df.groupBy(
        "location_id", "location_name", "location_type"
    ).agg(
        F.sum(LABEL_COLUMN).alias("actual_qty"),
        F.sum(PREDICTION_COLUMN).alias("predicted_qty"),
    )
    show_result(
        by_location.withColumn(
            "aggregate_absolute_error",
            F.abs(F.col("actual_qty") - F.col("predicted_qty")),
        ).orderBy("location_type", "location_name")
    )

    by_demand_band = (
        predictions_df.withColumn(
            "demand_band",
            F.when(F.col(LABEL_COLUMN) == 0, F.lit("0"))
            .when(F.col(LABEL_COLUMN) <= 5, F.lit("(0,5]"))
            .otherwise(F.lit(">5")),
        )
        .groupBy("demand_band")
        .agg(
            F.count(F.lit(1)).alias("rows"),
            F.sum(LABEL_COLUMN).alias("actual_qty"),
            F.sum(PREDICTION_COLUMN).alias("predicted_qty"),
            F.avg(PREDICTION_COLUMN).alias("predicted_avg_per_row"),
            F.sum(F.abs(F.col(LABEL_COLUMN) - F.col(PREDICTION_COLUMN))).alias(
                "absolute_error"
            ),
        )
    )
    show_result(by_demand_band.orderBy("demand_band"))

# COMMAND ----------

print(json.dumps(backtest_result, sort_keys=True))
try:
    dbutils.notebook.exit(json.dumps(backtest_result, sort_keys=True))  # noqa: F821
except NameError:
    pass
