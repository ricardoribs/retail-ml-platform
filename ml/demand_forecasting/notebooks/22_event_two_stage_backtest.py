# Databricks notebook source
# ruff: noqa: E402, F821

"""Experimental occurrence-and-quantity backtest at the SKU-store grain.

Does not write tables or change the job. Each fold freezes signals at the first month
of the holdout; months 2 and 3 use only the known calendar."""

from __future__ import annotations

import json
import math
import sys
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.regression import GBTRegressor
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
    raise RuntimeError("Project root not found")
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ml.demand_forecasting.src.config import (
    FORECAST_CONFIG,
    INTERMEDIATE_SCHEMA,
    fq_table,
)
from ml.demand_forecasting.src.hierarchical import evaluate_forecasts
from ml.demand_forecasting.src.intermittent_bayes import build_hurdle_forecasts
from ml.demand_forecasting.src.metrics import calculate_metrics
from ml.demand_forecasting.src.modeling import build_temporal_split, shift_months

FEATURE_TABLE = fq_table(INTERMEDIATE_SCHEMA, "int_forecast_event_features")
KEYS = ("product_variant_id", "location_id")
NUMERIC = (
    "last_observed_unit_price_brl",
    "previous_month_demand_qty",
    "positive_months_last_3",
    "positive_months_last_12",
    "sales_days_last_3",
    "orders_last_3",
    "has_price_history",
    "month_number",
)
FOLDS = (date(2025, 11, 30), date(2026, 8, 31))


def show_result(df: DataFrame) -> None:
    try:
        display(df)
    except NameError:
        df.show(200, truncate=False)


def prepare(df: DataFrame) -> DataFrame:
    """Impute features only; current demand remains a label."""
    result = df.withColumn("demand_qty", F.col("demand_qty").cast("double"))
    for column in NUMERIC:
        result = result.withColumn(
            column, F.coalesce(F.col(column).cast("double"), F.lit(0.0))
        )
    return result.withColumn(
        "occurrence", F.when(F.col("demand_qty") > 0, 1.0).otherwise(0.0)
    )


if not spark.catalog.tableExists(FEATURE_TABLE):
    raise ValueError(
        f"Run dbt build --select {FEATURE_TABLE.split('.')[-1]} first"
    )

history = spark.table(FEATURE_TABLE).filter(F.col("location_type") == "store")
if history.groupBy("demand_month", *KEYS).count().filter("count != 1").count():
    raise ValueError("The feature table has duplicate keys")
if history.filter(F.col("demand_qty").isNull() | (F.col("demand_qty") < 0)).count():
    raise ValueError("Labels are null or negative")

metric_rows = []
monthly_metric_rows = []
series_metric_rows = []
detail_rows = []
for cutoff in FOLDS:
    if cutoff > FORECAST_CONFIG.cutoff_date:
        raise ValueError(f"Cutoff is after the execution's closed month: {cutoff}")
    split = build_temporal_split(cutoff, test_months=3)
    targets = tuple(shift_months(split.test_start, i) for i in range(3))
    before = history.filter(F.col("demand_month") <= F.lit(split.train_end))
    eligible = (
        before.groupBy(*KEYS)
        .agg(
            F.count("demand_month").alias("months"),
            F.sum(F.when(F.col("demand_qty") > 0, 1).otherwise(0)).alias("positive"),
        )
        .filter(
            (F.col("months") >= FORECAST_CONFIG.min_history_months)
            & (F.col("positive") >= FORECAST_CONFIG.min_positive_demand_months)
        )
        .select(*KEYS)
    )
    cohort = history.join(eligible, list(KEYS), "inner")
    train = prepare(
        cohort.filter(
            (F.col("demand_month") <= F.lit(split.train_end))
            & (F.col("months_available_before_target") >= 6)
        )
    )
    actual_df = cohort.filter(
        (F.col("demand_month") >= F.lit(split.test_start))
        & (F.col("demand_month") <= F.lit(split.test_end))
    )
    origin_df = cohort.filter(F.col("demand_month") == F.lit(split.test_start))
    series_count = eligible.count()
    if not series_count or train.count() == 0 or origin_df.count() != series_count:
        raise ValueError(f"Incomplete cohort or training set at cutoff {cutoff}")
    if actual_df.count() != series_count * len(targets):
        raise ValueError(f"Incomplete holdout at cutoff {cutoff}")

    assembler = VectorAssembler(inputCols=list(NUMERIC), outputCol="features")
    train_vectors = assembler.transform(train)
    occurrence_model = LogisticRegression(
        featuresCol="features", labelCol="occurrence", maxIter=60, regParam=0.1
    ).fit(train_vectors)
    positive_vectors = train_vectors.filter(F.col("occurrence") == 1.0)
    if positive_vectors.count() < 2:
        raise ValueError(f"Too few positive sales for training: {cutoff}")
    quantity_model = GBTRegressor(
        featuresCol="features",
        labelCol="demand_qty",
        maxIter=35,
        maxDepth=4,
        seed=42,
    ).fit(positive_vectors)

    # The first target-month row contains only signals through the end of training.
    # Never read features from holdout months 2 and 3.
    frozen = prepare(origin_df).select(*KEYS, *NUMERIC)
    candidate = {}
    for target in targets:
        future = frozen.withColumn("month_number", F.lit(float(target.month)))
        vectors = assembler.transform(future)
        occurrence = occurrence_model.transform(vectors).select(
            *KEYS, vector_to_array("probability")[1].alias("sale_probability")
        )
        quantity = quantity_model.transform(vectors).select(
            *KEYS, F.greatest(F.col("prediction"), F.lit(0.0)).alias("positive_qty")
        )
        for row in occurrence.join(quantity, list(KEYS)).collect():
            key = (target, row[KEYS[0]], row[KEYS[1]])
            probability = float(row["sale_probability"])
            conditional = float(row["positive_qty"])
            prediction = probability * conditional
            if not all(
                math.isfinite(v) and v >= 0
                for v in (probability, conditional, prediction)
            ):
                raise ValueError(f"Invalid forecast: {key}")
            candidate[key] = prediction
            detail_rows.append(
                {
                    "cutoff": str(cutoff),
                    "forecast_month": str(target),
                    "product_variant_id": key[1],
                    "location_id": key[2],
                    "sale_probability": probability,
                    "qty_if_sale": conditional,
                    "forecast_qty": prediction,
                }
            )

    actual = {
        (row["demand_month"], row[KEYS[0]], row[KEYS[1]]): float(row["demand_qty"])
        for row in actual_df.select("demand_month", *KEYS, "demand_qty").collect()
    }
    history_rows = (
        before.join(eligible, list(KEYS))
        .select("demand_month", *KEYS, "category_name", "demand_qty")
        .collect()
    )
    groups = {}
    series_history = {}
    for row in history_rows:
        series = (row[KEYS[0]], row[KEYS[1]])
        groups[series] = (row[KEYS[1]], row["category_name"] or "__missing__")
        series_history.setdefault(series, {})[row["demand_month"]] = float(
            row["demand_qty"]
        )
    baselines = build_hurdle_forecasts(
        series_groups=groups, series_history=series_history, target_months=targets
    )
    predictions = {
        "two_stage_price_frequency": candidate,
        "zero": baselines["zero"],
        "ma3": baselines["ma3_frozen"],
        "seasonal": baselines["seasonal_series"],
    }
    for name, values in predictions.items():
        metrics = evaluate_forecasts(actual, values, groups)["sku_location"]
        metric_rows.append(
            {
                "cutoff": str(cutoff),
                "method": name,
                "series": series_count,
                "mae": float(metrics["mae"]),
                "wape": float(metrics["wape"]),
                "actual_total": sum(actual.values()),
                "forecast_total": sum(values.values()),
            }
        )
        for target in targets:
            month_actual = {
                key: value for key, value in actual.items() if key[0] == target
            }
            month_values = {key: values[key] for key in month_actual}
            month_metrics = evaluate_forecasts(month_actual, month_values, groups)[
                "sku_location"
            ]
            monthly_metric_rows.append(
                {
                    "cutoff": str(cutoff),
                    "method": name,
                    "series": series_count,
                    "forecast_month": str(target),
                    "mae": float(month_metrics["mae"]),
                    "wape": float(month_metrics["wape"]),
                    "actual_total": sum(month_actual.values()),
                    "forecast_total": sum(month_values.values()),
                }
            )
        for product_id, location_id in groups:
            keys = [(month, product_id, location_id) for month in targets]
            series_metrics = calculate_metrics(
                [actual[key] for key in keys], [values[key] for key in keys]
            )
            series_metric_rows.append(
                {
                    "cutoff": str(cutoff),
                    "method": name,
                    "product_variant_id": product_id,
                    "location_id": location_id,
                    "mae": float(series_metrics["mae"]),
                    "wape": (
                        float(series_metrics["wape"])
                        if math.isfinite(series_metrics["wape"])
                        else None
                    ),
                }
            )

summary = spark.createDataFrame(metric_rows)
show_result(summary.orderBy("cutoff", "method"))
show_result(
    spark.createDataFrame(monthly_metric_rows).orderBy(
        "cutoff", "forecast_month", "method"
    )
)
show_result(
    spark.createDataFrame(series_metric_rows).orderBy("cutoff", "method", *KEYS)
)
for cutoff in FOLDS:
    rows = [row for row in metric_rows if row["cutoff"] == str(cutoff)]
    candidate_wape = next(
        row["wape"] for row in rows if row["method"] == "two_stage_price_frequency"
    )
    best_baseline = min(
        row["wape"] for row in rows if row["method"] != "two_stage_price_frequency"
    )
    months_pass = all(
        next(
            row["wape"]
            for row in monthly_metric_rows
            if row["cutoff"] == str(cutoff)
            and row["forecast_month"] == str(month)
            and row["method"] == "two_stage_price_frequency"
        )
        < min(
            row["wape"]
            for row in monthly_metric_rows
            if row["cutoff"] == str(cutoff)
            and row["forecast_month"] == str(month)
            and row["method"] != "two_stage_price_frequency"
        )
        for month in (
            shift_months(build_temporal_split(cutoff, 3).test_start, i)
            for i in range(3)
        )
    )
    print(
        f"{cutoff}: outperforms baselines overall and in every month = "
        f"{candidate_wape < best_baseline and months_pass}"
    )

# Inspectable evidence; no automatic publication.
show_result(
    spark.createDataFrame(detail_rows).orderBy("cutoff", "forecast_month", *KEYS)
)

# Compact output for the Jobs API; the tables above remain the detailed
# evidence. This notebook still does not publish a forecast or model.
backtest_result = {
    "folds": metric_rows,
    "monthly": monthly_metric_rows,
}
try:
    dbutils.notebook.exit(json.dumps(backtest_result, sort_keys=True))  # noqa: F821
except NameError:
    print(json.dumps(backtest_result, sort_keys=True))
