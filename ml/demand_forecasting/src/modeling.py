"""Temporal training and evaluation for demand-forecast regression models.

This module contains the reusable forecast contract: selecting eligible
series without looking at the test window, preparing features, building
Spark ML pipelines, and evaluating with the project's official metrics.

`pyspark` imports are local to functions so pure configuration and date tests
remain runnable in CI, which installs only numpy and pytest. Model training
runs in Databricks."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

import numpy as np

from ml.demand_forecasting.src.metrics import calculate_metrics
from ml.demand_forecasting.src.recursive import (
    FROZEN_SIGNAL_COLUMNS,
    SeriesKey,
    SeriesState,
    recursive_forecast,
)

if TYPE_CHECKING:
    from pyspark.ml import PipelineModel
    from pyspark.sql import DataFrame


LABEL_COLUMN = "demand_qty"
PREDICTION_COLUMN = "prediction"
# v3 forces a refresh of persisted metrics: the global champion now includes
# every evaluated candidate, including the zero-demand baseline.
RECURSIVE_MODEL_VERSION = "regression_v3_recursive_overall"
SERIES_KEY_COLUMNS: tuple[str, ...] = (
    "product_variant_id",
    "location_id",
)

# Features observed during the target month (revenue, orders, and sales days) are
# excluded from training because they are known only after the forecast month.
NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = (
    "lag_1_qty",
    "lag_2_qty",
    "lag_3_qty",
    "rolling_3m_avg_qty",
    "rolling_6m_avg_qty",
    "month_number",
    "quarter_number",
    "year_number",
    *FROZEN_SIGNAL_COLUMNS,
)
CATEGORICAL_FEATURE_COLUMNS: tuple[str, ...] = (
    "product_variant_id",
    "category_name",
    "location_id",
    "location_type",
    "state",
)
CATEGORICAL_MODEL_COLUMNS: tuple[str, ...] = tuple(
    f"{column}__category" for column in CATEGORICAL_FEATURE_COLUMNS
)


@dataclass(frozen=True)
class TemporalSplit:
    """Inclusive boundaries for the temporal training and test windows."""

    train_end: date
    test_start: date
    test_end: date


@dataclass(frozen=True)
class RegressionModelSpec:
    """Versionable configuration for a regression family."""

    model_name: str
    family: str
    parameters: dict[str, float | int | str]


REGRESSION_MODEL_SPECS: tuple[RegressionModelSpec, ...] = (
    RegressionModelSpec(
        model_name="ridge_regression",
        family="ridge",
        parameters={
            "elasticNetParam": 0.0,
            "maxIter": 100,
            "regParam": 0.1,
        },
    ),
    RegressionModelSpec(
        model_name="random_forest",
        family="random_forest",
        parameters={
            "maxDepth": 8,
            "numTrees": 50,
            "seed": 42,
        },
    ),
    RegressionModelSpec(
        model_name="gradient_boosted_trees",
        family="gradient_boosted_trees",
        parameters={
            "maxDepth": 5,
            "maxIter": 50,
            "seed": 42,
            "stepSize": 0.05,
        },
    ),
)

# Research candidate, excluded from the operational job until it passes backtests.
# WAPE is proportional to the sum of absolute errors; L1 loss tests whether it can
# reduce the positive average forecast that L2 loss assigns to months with no sales.
INTERMITTENT_MODEL_SPECS: tuple[RegressionModelSpec, ...] = (
    RegressionModelSpec(
        model_name="gradient_boosted_trees_l1",
        family="gradient_boosted_trees",
        parameters={
            "lossType": "absolute",
            "maxDepth": 5,
            "maxIter": 50,
            "seed": 42,
            "stepSize": 0.05,
        },
    ),
)


def shift_months(reference: date, months: int) -> date:
    """Shift a date by months and return the first day of the resulting month."""
    month_index = reference.month - 1 + months
    year = reference.year + month_index // 12
    month = month_index % 12 + 1
    return date(year, month, 1)


def build_temporal_split(cutoff_date: date, test_months: int) -> TemporalSplit:
    """Build historical training data and use the latest ``test_months`` as holdout."""
    if test_months <= 0:
        raise ValueError("test_months must be greater than zero")

    cutoff_month = cutoff_date.replace(day=1)
    test_start = shift_months(cutoff_month, -(test_months - 1))
    train_end = test_start - timedelta(days=1)
    return TemporalSplit(
        train_end=train_end,
        test_start=test_start,
        test_end=cutoff_date,
    )


def prepare_model_features(
    feature_df: DataFrame,
    include_label: bool = True,
) -> DataFrame:
    """Prepare numeric and categorical columns for training or prediction."""
    from pyspark.sql import functions as F

    prepared_df = feature_df
    numeric_columns = list(NUMERIC_FEATURE_COLUMNS)
    if include_label and LABEL_COLUMN in prepared_df.columns:
        numeric_columns.append(LABEL_COLUMN)

    for column in numeric_columns:
        prepared_df = prepared_df.withColumn(column, F.col(column).cast("double"))

    for source_column, model_column in zip(
        CATEGORICAL_FEATURE_COLUMNS,
        CATEGORICAL_MODEL_COLUMNS,
        strict=True,
    ):
        prepared_df = prepared_df.withColumn(
            model_column,
            F.coalesce(F.col(source_column).cast("string"), F.lit("__missing__")),
        )

    return prepared_df


def prepare_temporal_frames(
    feature_matrix_df: DataFrame,
    split: TemporalSplit,
    min_history_months: int,
    min_positive_demand_months: int,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    """Select eligible training series and return prepared training/test frames.

    Recalculate eligibility using only observations through ``train_end``. This
    prevents demand observed in the test window from determining which series
    enter the model.
    """
    from pyspark.sql import functions as F

    history_before_test_df = feature_matrix_df.filter(
        F.col("demand_month") <= F.lit(split.train_end)
    )

    eligible_series_df = (
        history_before_test_df.groupBy(*SERIES_KEY_COLUMNS)
        .agg(
            F.countDistinct("demand_month").alias("history_months_at_train_end"),
            F.sum(
                F.when(F.col(LABEL_COLUMN) > F.lit(0), F.lit(1)).otherwise(F.lit(0))
            ).alias("positive_months_at_train_end"),
        )
        .filter(
            (F.col("history_months_at_train_end") >= F.lit(min_history_months))
            & (
                F.col("positive_months_at_train_end")
                >= F.lit(min_positive_demand_months)
            )
        )
    )

    prepared_df = feature_matrix_df.filter(
        F.col("demand_month") <= F.lit(split.test_end)
    ).join(
        eligible_series_df.select(*SERIES_KEY_COLUMNS),
        list(SERIES_KEY_COLUMNS),
        "inner",
    )

    prepared_df = prepare_model_features(prepared_df)

    train_df = prepared_df.filter(
        (F.col("demand_month") <= F.lit(split.train_end))
        & (F.col("months_available_before_target") >= F.lit(min_history_months))
    )
    test_df = prepared_df.filter(
        (F.col("demand_month") >= F.lit(split.test_start))
        & (F.col("demand_month") <= F.lit(split.test_end))
    )

    return eligible_series_df, train_df, test_df


def build_model_pipeline(spec: RegressionModelSpec) -> Any:
    """Build a complete Spark ML pipeline for a model specification."""
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import (
        Imputer,
        OneHotEncoder,
        StringIndexer,
        VectorAssembler,
    )
    from pyspark.ml.regression import (
        GBTRegressor,
        LinearRegression,
        RandomForestRegressor,
    )

    indexed_columns = tuple(
        f"{column}__indexed" for column in CATEGORICAL_MODEL_COLUMNS
    )
    encoded_columns = tuple(
        f"{column}__encoded" for column in CATEGORICAL_MODEL_COLUMNS
    )
    imputed_columns = tuple(f"{column}__imputed" for column in NUMERIC_FEATURE_COLUMNS)

    indexers = [
        StringIndexer(
            inputCol=input_column,
            outputCol=output_column,
            handleInvalid="keep",
            stringOrderType="alphabetAsc",
        )
        for input_column, output_column in zip(
            CATEGORICAL_MODEL_COLUMNS,
            indexed_columns,
            strict=True,
        )
    ]
    encoder = OneHotEncoder(
        inputCols=list(indexed_columns),
        outputCols=list(encoded_columns),
        handleInvalid="keep",
        dropLast=True,
    )
    imputer = Imputer(
        inputCols=list(NUMERIC_FEATURE_COLUMNS),
        outputCols=list(imputed_columns),
        strategy="median",
    )
    assembler = VectorAssembler(
        inputCols=[*imputed_columns, *encoded_columns],
        outputCol="features",
        handleInvalid="error",
    )

    common_parameters: dict[str, Any] = {
        "featuresCol": "features",
        "labelCol": LABEL_COLUMN,
        "predictionCol": PREDICTION_COLUMN,
    }
    model_parameters = {**common_parameters, **spec.parameters}

    if spec.family == "ridge":
        estimator = LinearRegression(**model_parameters)
    elif spec.family == "random_forest":
        estimator = RandomForestRegressor(**model_parameters)
    elif spec.family == "gradient_boosted_trees":
        estimator = GBTRegressor(**model_parameters)
    else:
        raise ValueError(f"Unsupported model family: {spec.family}")

    return Pipeline(stages=[*indexers, encoder, imputer, assembler, estimator])


def collect_backtest_history(
    feature_matrix_df: DataFrame,
    test_df: DataFrame,
    train_end: date,
) -> dict[SeriesKey, SeriesState]:
    """Get twelve known months for features and the seasonal baseline."""
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    tested_keys_df = test_df.select(*SERIES_KEY_COLUMNS).distinct()
    recent_window = Window.partitionBy(*SERIES_KEY_COLUMNS).orderBy(
        F.col("demand_month").desc()
    )
    recent_df = (
        feature_matrix_df.filter(F.col("demand_month") <= F.lit(train_end))
        .join(tested_keys_df, list(SERIES_KEY_COLUMNS), "inner")
        .withColumn("_row_number", F.row_number().over(recent_window))
        .filter(F.col("_row_number") <= F.lit(12))
    )
    history_columns = (
        "demand_month",
        LABEL_COLUMN,
        *CATEGORICAL_FEATURE_COLUMNS,
    )
    states: dict[SeriesKey, SeriesState] = {}
    for row in recent_df.select(*history_columns).collect():
        key = (row["product_variant_id"], row["location_id"])
        if key not in states:
            states[key] = {"static": {}, "history": []}
        states[key]["history"].append(
            (row["demand_month"], float(row[LABEL_COLUMN] or 0.0))
        )
        if (
            not states[key]["static"]
            or row["demand_month"] > states[key]["static_month"]
        ):
            states[key]["static"] = {
                column: row[column] for column in CATEGORICAL_FEATURE_COLUMNS
            }
            states[key]["static_month"] = row["demand_month"]

    first_test_window = Window.partitionBy(*SERIES_KEY_COLUMNS).orderBy("demand_month")
    origin_signals_df = (
        test_df.withColumn("_row_number", F.row_number().over(first_test_window))
        .filter(F.col("_row_number") == F.lit(1))
        .select(*SERIES_KEY_COLUMNS, *FROZEN_SIGNAL_COLUMNS)
    )
    for row in origin_signals_df.collect():
        key = (row["product_variant_id"], row["location_id"])
        if key not in states:
            raise ValueError("A test series has no history before the cutoff")
        states[key]["static"].update(
            {
                column: float(row[column]) if row[column] is not None else None
                for column in FROZEN_SIGNAL_COLUMNS
            }
        )

    expected_keys = {
        (row["product_variant_id"], row["location_id"])
        for row in tested_keys_df.collect()
    }
    if set(states) != expected_keys:
        raise ValueError("Test series are missing history before the cutoff")
    for state in states.values():
        state["history"].sort(key=lambda item: item[0])
        del state["static_month"]
    return states


def recursive_prediction_frame(
    fitted_model: PipelineModel,
    test_df: DataFrame,
    history_states: dict[SeriesKey, SeriesState],
    target_months: tuple[date, ...],
) -> DataFrame:
    """Apply the model across the horizon as in the operational forecast."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import DateType, DoubleType, StructField, StructType

    spark = test_df.sparkSession

    def predict_batch(inputs: list[dict[str, Any]]) -> dict[SeriesKey, float]:
        model_input_df = spark.createDataFrame(inputs)
        prepared_df = prepare_model_features(model_input_df, include_label=False)
        prediction_rows = (
            fitted_model.transform(prepared_df)
            .select(
                F.col("_key_product_variant_id").alias("product_variant_id"),
                F.col("_key_location_id").alias("location_id"),
                F.greatest(F.col(PREDICTION_COLUMN), F.lit(0.0)).alias(
                    PREDICTION_COLUMN
                ),
            )
            .collect()
        )
        return {
            (row["product_variant_id"], row["location_id"]): float(
                row[PREDICTION_COLUMN]
            )
            for row in prediction_rows
        }

    prediction_records = recursive_forecast(
        history_states, target_months, predict_batch
    )
    prediction_schema = StructType(
        [
            StructField("demand_month", DateType(), False),
            *(
                StructField(column, test_df.schema[column].dataType, False)
                for column in SERIES_KEY_COLUMNS
            ),
            StructField(PREDICTION_COLUMN, DoubleType(), False),
        ]
    )
    prediction_rows = [
        (
            row["demand_month"],
            row["product_variant_id"],
            row["location_id"],
            row[PREDICTION_COLUMN],
        )
        for row in prediction_records
    ]
    prediction_df = spark.createDataFrame(prediction_rows, schema=prediction_schema)
    result_df = test_df.join(
        prediction_df, ["demand_month", *SERIES_KEY_COLUMNS], "left"
    )
    if result_df.filter(F.col(PREDICTION_COLUMN).isNull()).limit(1).count():
        raise ValueError("Recursive forecasts did not cover every test row")
    return result_df


def train_and_evaluate_models(
    train_df: DataFrame,
    test_df: DataFrame,
    specs: tuple[RegressionModelSpec, ...] = REGRESSION_MODEL_SPECS,
    *,
    history_states: dict[SeriesKey, SeriesState] | None = None,
    target_months: tuple[date, ...] | None = None,
    legacy_one_step: bool = False,
) -> tuple[dict[str, PipelineModel], dict[str, DataFrame], list[dict[str, Any]]]:
    """Train and evaluate recursively; one-step evaluation is for V1 audits only."""
    from pyspark.sql import functions as F

    training_rows = train_df.count()
    test_rows = test_df.count()
    if training_rows == 0:
        raise ValueError("The training set is empty")
    if test_rows == 0:
        raise ValueError("The test set is empty")
    if (history_states is None) != (target_months is None):
        raise ValueError("Provide both history_states and target_months")
    if (legacy_one_step and history_states is not None) or (
        not legacy_one_step and history_states is None
    ):
        raise ValueError(
            "Explicitly choose recursive evaluation or legacy_one_step"
        )

    trained_models: dict[str, PipelineModel] = {}
    prediction_frames: dict[str, DataFrame] = {}
    metric_rows: list[dict[str, Any]] = []

    for spec in specs:
        print(f"Training {spec.model_name} with parameters {spec.parameters}")
        started_at = time.perf_counter()
        fitted_model = build_model_pipeline(spec).fit(train_df)
        training_seconds = time.perf_counter() - started_at

        if history_states is None:
            predictions_df = fitted_model.transform(test_df).withColumn(
                PREDICTION_COLUMN,
                F.greatest(F.col(PREDICTION_COLUMN), F.lit(0.0)),
            )
        else:
            predictions_df = recursive_prediction_frame(
                fitted_model, test_df, history_states, target_months
            )
        evaluation_rows = predictions_df.select(
            LABEL_COLUMN,
            PREDICTION_COLUMN,
        ).collect()
        actual = np.asarray(
            [row[LABEL_COLUMN] for row in evaluation_rows],
            dtype=float,
        )
        predicted = np.asarray(
            [row[PREDICTION_COLUMN] for row in evaluation_rows],
            dtype=float,
        )
        metrics = calculate_metrics(actual, predicted)

        trained_models[spec.model_name] = fitted_model
        prediction_frames[spec.model_name] = predictions_df
        metric_rows.append(
            {
                "model_name": spec.model_name,
                "model_family": spec.family,
                **metrics,
                "fallback_rows": 0,
                "parameters_json": json.dumps(spec.parameters, sort_keys=True),
                "training_rows": training_rows,
                "training_seconds": float(training_seconds),
            }
        )
        print(
            f"{spec.model_name} completed in {training_seconds:.2f}s "
            f"(WAPE={metrics['wape']:.4f}, MAE={metrics['mae']:.4f})"
        )

    return trained_models, prediction_frames, metric_rows


def select_best_model(metric_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Select the lowest WAPE, using MAE and RMSE as tie-breakers."""
    valid_rows = [
        row
        for row in metric_rows
        if math.isfinite(float(row["wape"]))
        and math.isfinite(float(row["mae"]))
        and math.isfinite(float(row["rmse"]))
    ]
    if not valid_rows:
        raise ValueError("No model has finite metrics for selection")

    return min(
        valid_rows,
        key=lambda row: (
            float(row["wape"]),
            float(row["mae"]),
            float(row["rmse"]),
            str(row["model_name"]),
        ),
    )


def select_deployable_regression(
    metric_rows: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Require regression to beat zero, moving-average, and seasonal baselines.

    This is only a quality floor, not a guarantee of operational usefulness.
    """
    regression_names = {spec.model_name for spec in REGRESSION_MODEL_SPECS}
    reference_names = {
        "baseline_zero",
        "baseline_ma3",
        "baseline_ma6",
        "baseline_seasonal",
    }
    model_names = [str(row["model_name"]) for row in metric_rows]
    if len(model_names) != len(set(model_names)):
        raise ValueError("Duplicate models were found in the quality comparison")
    rows_by_name = {str(row["model_name"]): row for row in metric_rows}
    if not (regression_names | reference_names).issubset(rows_by_name):
        raise ValueError(
            "Incomplete quality comparison: models or baselines are missing"
        )

    compared_rows = [rows_by_name[name] for name in regression_names | reference_names]
    if any(
        row["wape"] is None or not math.isfinite(float(row["wape"]))
        for row in compared_rows
    ):
        raise ValueError("WAPE is missing or non-finite in the run metrics")

    best_regression = min(
        (rows_by_name[name] for name in regression_names),
        key=lambda row: (
            float(row["wape"]),
            float(row["mae"]),
            float(row["rmse"]),
            str(row["model_name"]),
        ),
    )
    best_reference = min(
        (rows_by_name[name] for name in reference_names),
        key=lambda row: float(row["wape"]),
    )
    if float(best_regression["wape"]) >= float(best_reference["wape"]):
        raise ValueError(
            "Forecast not published: no regression model outperformed the benchmark "
            f"recursiva {best_reference['model_name']} "
            f"(WAPE {best_regression['wape']:.2f}% vs "
            f"{best_reference['wape']:.2f}%)."
        )
    return best_regression, best_reference
