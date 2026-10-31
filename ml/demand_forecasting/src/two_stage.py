"""Experimental two-stage candidate for intermittent monthly demand.

It trains sale occurrence on all rows and conditional quantity only on
positive months. The backtest applies both stages recursively without using
test-month actuals as features. Nothing in this module persists models or
tables."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np

from ml.demand_forecasting.src.metrics import calculate_metrics
from ml.demand_forecasting.src.modeling import (
    CATEGORICAL_FEATURE_COLUMNS,
    LABEL_COLUMN,
    NUMERIC_FEATURE_COLUMNS,
    PREDICTION_COLUMN,
    SERIES_KEY_COLUMNS,
    prepare_model_features,
)
from ml.demand_forecasting.src.recursive import (
    SeriesKey,
    SeriesState,
    recursive_forecast,
)

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


MODEL_NAME = "two_stage_gbt"
EXPECTED_MODEL_NAME = f"{MODEL_NAME}_expected"
THRESHOLDS = (0.10, 0.25, 0.50, 0.75)
TREE_PARAMETERS = {"maxDepth": 5, "maxIter": 50, "seed": 42, "stepSize": 0.05}


def calculate_occurrence_metrics(
    actual_quantity: Sequence[float],
    occurrence_probability: Sequence[float],
    reference_probability: float,
    decision_threshold: float = 0.5,
) -> dict[str, float]:
    """Measure the occurrence stage separately from the quantity sold.

    The reference probability comes from training data only. The comparison's
    Brier score therefore does not use the holdout's observed incidence as a
    forecast. AUC and average precision measure discrimination; Brier also
    measures probability calibration.
    """
    actual = np.asarray(actual_quantity, dtype=float)
    probability = np.asarray(occurrence_probability, dtype=float)
    if actual.ndim != 1 or probability.ndim != 1 or len(actual) != len(probability):
        raise ValueError("Quantity and probability arrays must have the same size")
    if len(actual) == 0:
        raise ValueError("Occurrence evaluation is empty")
    if not np.isfinite(actual).all() or (actual < 0).any():
        raise ValueError("Invalid actual quantity")
    if (
        not np.isfinite(probability).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise ValueError("Invalid occurrence probability")
    if not math.isfinite(reference_probability) or not 0 <= reference_probability <= 1:
        raise ValueError("Invalid reference probability")
    if not math.isfinite(decision_threshold) or not 0 < decision_threshold < 1:
        raise ValueError("Decision threshold must be between zero and one")

    event = (actual > 0).astype(float)
    positives = int(event.sum())
    negatives = len(event) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("The holdout must contain months with and without sales")

    brier_score = float(np.mean((probability - event) ** 2))
    reference_brier_score = float(np.mean((reference_probability - event) ** 2))

    # Average ranks make ROC-AUC deterministic in the presence of tied scores.
    sorted_indexes = np.argsort(probability, kind="stable")
    ranks = np.empty(len(probability), dtype=float)
    cursor = 0
    while cursor < len(sorted_indexes):
        end = cursor + 1
        score = probability[sorted_indexes[cursor]]
        while end < len(sorted_indexes) and probability[sorted_indexes[end]] == score:
            end += 1
        ranks[sorted_indexes[cursor:end]] = (cursor + 1 + end) / 2
        cursor = end
    roc_auc = float(
        (ranks[event == 1].sum() - positives * (positives + 1) / 2)
        / (positives * negatives)
    )

    # AP is evaluated at score groups so equal probabilities are treated fairly.
    descending_indexes = np.argsort(-probability, kind="stable")
    true_positives = 0
    processed = 0
    average_precision = 0.0
    cursor = 0
    while cursor < len(descending_indexes):
        end = cursor + 1
        score = probability[descending_indexes[cursor]]
        while (
            end < len(descending_indexes)
            and probability[descending_indexes[end]] == score
        ):
            end += 1
        group_events = int(event[descending_indexes[cursor:end]].sum())
        true_positives += group_events
        processed += end - cursor
        average_precision += (group_events / positives) * (true_positives / processed)
        cursor = end

    predicted_event = probability >= decision_threshold
    true_positive = int(np.logical_and(predicted_event, event == 1).sum())
    false_positive = int(np.logical_and(predicted_event, event == 0).sum())
    false_negative = int(np.logical_and(~predicted_event, event == 1).sum())
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = true_positive / (true_positive + false_negative)

    return {
        "event_rate": positives / len(event),
        "brier_score": brier_score,
        "reference_brier_score": reference_brier_score,
        "brier_skill_score": 1.0 - brier_score / reference_brier_score,
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "decision_threshold": decision_threshold,
        "decision_precision": precision,
        "decision_recall": recall,
        "decision_positive_rate": float(predicted_event.mean()),
    }


def gated_quantity(
    occurrence_probability: float,
    positive_quantity: float,
    threshold: float,
) -> float:
    """Return zero or the conditional quantity, never a negative value."""
    if not math.isfinite(threshold) or not 0 < threshold < 1:
        raise ValueError("Occurrence threshold must be between zero and one")
    if not math.isfinite(occurrence_probability) or not (
        0 <= occurrence_probability <= 1
    ):
        raise ValueError("Invalid occurrence probability")
    if not math.isfinite(positive_quantity):
        raise ValueError("Invalid conditional quantity")
    if occurrence_probability < threshold:
        return 0.0
    return max(positive_quantity, 0.0)


def expected_quantity(
    occurrence_probability: float,
    positive_quantity: float,
) -> float:
    """Combine occurrence and magnitude into a single demand forecast.

    The final quantity is ``P(sale) * E[quantity | sale]``. Unlike a
    classification cutoff, this rule preserves occurrence uncertainty and can be
    compared directly with actual monthly quantity.
    """
    if not math.isfinite(occurrence_probability) or not (
        0 <= occurrence_probability <= 1
    ):
        raise ValueError("Invalid occurrence probability")
    if not math.isfinite(positive_quantity):
        raise ValueError("Invalid conditional quantity")
    return occurrence_probability * max(positive_quantity, 0.0)


def _build_preprocessing_pipeline() -> Any:
    """Mirror the operational regression features without including the target."""
    from pyspark.ml import Pipeline
    from pyspark.ml.feature import (
        Imputer,
        OneHotEncoder,
        StringIndexer,
        VectorAssembler,
    )

    indexed_columns = [
        f"{column}__category__indexed" for column in CATEGORICAL_FEATURE_COLUMNS
    ]
    encoded_columns = [
        f"{column}__category__encoded" for column in CATEGORICAL_FEATURE_COLUMNS
    ]
    imputed_columns = [f"{column}__imputed" for column in NUMERIC_FEATURE_COLUMNS]
    indexers = [
        StringIndexer(
            inputCol=f"{column}__category",
            outputCol=output_column,
            handleInvalid="keep",
            stringOrderType="alphabetAsc",
        )
        for column, output_column in zip(
            CATEGORICAL_FEATURE_COLUMNS, indexed_columns, strict=True
        )
    ]
    encoder = OneHotEncoder(
        inputCols=indexed_columns,
        outputCols=encoded_columns,
        handleInvalid="keep",
        dropLast=True,
    )
    imputer = Imputer(
        inputCols=list(NUMERIC_FEATURE_COLUMNS),
        outputCols=imputed_columns,
        strategy="median",
    )
    assembler = VectorAssembler(
        inputCols=[*imputed_columns, *encoded_columns],
        outputCol="features",
        handleInvalid="error",
    )
    return Pipeline(stages=[*indexers, encoder, imputer, assembler])


def _fit_models(train_df: DataFrame) -> tuple[Any, Any, Any, int, float, float]:
    from pyspark.ml.classification import GBTClassifier
    from pyspark.ml.regression import GBTRegressor
    from pyspark.sql import functions as F

    class_counts = train_df.agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.when(F.col(LABEL_COLUMN) > 0, 1).otherwise(0)).alias("positive"),
        F.sum(F.when(F.col(LABEL_COLUMN) == 0, 1).otherwise(0)).alias("zero"),
    ).first()
    if class_counts["rows"] == 0:
        raise ValueError("Two-stage training set is empty")
    if class_counts["rows"] != class_counts["positive"] + class_counts["zero"]:
        raise ValueError("Two-stage training contains negative or null demand")
    if class_counts["positive"] == 0 or class_counts["zero"] == 0:
        raise ValueError("Two-stage training requires both positive and zero months")
    training_rows = int(class_counts["rows"])

    started_at = time.perf_counter()
    preprocessing_model = _build_preprocessing_pipeline().fit(train_df)
    features_df = preprocessing_model.transform(train_df)
    occurrence_label_df = features_df.withColumn(
        "_has_demand_label", (F.col(LABEL_COLUMN) > 0).cast("double")
    )
    classifier_estimator = GBTClassifier(
        featuresCol="features",
        labelCol="_has_demand_label",
        predictionCol="_occurrence_class_prediction",
        **TREE_PARAMETERS,
    )
    classifier_estimator.setProbabilityCol("_occurrence_probability")
    classifier_estimator.setRawPredictionCol("_occurrence_raw_prediction")
    classifier = classifier_estimator.fit(occurrence_label_df)
    magnitude = GBTRegressor(
        featuresCol="features",
        labelCol=LABEL_COLUMN,
        predictionCol="_positive_qty_prediction",
        lossType="squared",
        **TREE_PARAMETERS,
    ).fit(features_df.filter(F.col(LABEL_COLUMN) > 0))
    training_seconds = time.perf_counter() - started_at
    occurrence_rate = float(class_counts["positive"] / class_counts["rows"])
    return (
        preprocessing_model,
        classifier,
        magnitude,
        training_rows,
        training_seconds,
        occurrence_rate,
    )


def _recursive_prediction_frame(
    test_df: DataFrame,
    history_states: Mapping[SeriesKey, SeriesState],
    target_months: Sequence[date],
    preprocessing_model: Any,
    classifier: Any,
    magnitude: Any,
    threshold: float | None,
) -> DataFrame:
    from pyspark.sql import functions as F
    from pyspark.sql.types import DateType, DoubleType, StructField, StructType

    spark = test_df.sparkSession
    diagnostics: dict[tuple[date, int, int], tuple[float, float]] = {}

    def predict_batch(inputs: list[dict[str, Any]]) -> dict[SeriesKey, float]:
        target_month = date(inputs[0]["year_number"], inputs[0]["month_number"], 1)
        input_df = spark.createDataFrame(inputs)
        features_df = preprocessing_model.transform(
            prepare_model_features(input_df, include_label=False)
        )
        scored_df = magnitude.transform(classifier.transform(features_df))
        scored_rows = scored_df.select(
            "_key_product_variant_id",
            "_key_location_id",
            "_occurrence_probability",
            "_positive_qty_prediction",
        ).collect()
        if len(scored_rows) != len(inputs):
            raise ValueError("Two-stage evaluation lost input rows")

        predictions: dict[SeriesKey, float] = {}
        for row in scored_rows:
            key = (row["_key_product_variant_id"], row["_key_location_id"])
            probability = float(row["_occurrence_probability"][1])
            positive_qty = float(row["_positive_qty_prediction"])
            predictions[key] = (
                expected_quantity(probability, positive_qty)
                if threshold is None
                else gated_quantity(probability, positive_qty, threshold)
            )
            diagnostics[(target_month, *key)] = (probability, positive_qty)
        if len(predictions) != len(inputs):
            raise ValueError("Two-stage evaluation duplicated series")
        return predictions

    prediction_records = recursive_forecast(
        history_states, target_months, predict_batch
    )
    schema = StructType(
        [
            StructField("demand_month", DateType(), False),
            *(
                StructField(column, test_df.schema[column].dataType, False)
                for column in SERIES_KEY_COLUMNS
            ),
            StructField(PREDICTION_COLUMN, DoubleType(), False),
            StructField("occurrence_probability", DoubleType(), False),
            StructField("positive_qty_prediction", DoubleType(), False),
        ]
    )
    prediction_rows = [
        (
            row["demand_month"],
            row["product_variant_id"],
            row["location_id"],
            row[PREDICTION_COLUMN],
            *diagnostics[
                (row["demand_month"], row["product_variant_id"], row["location_id"])
            ],
        )
        for row in prediction_records
    ]
    prediction_df = spark.createDataFrame(prediction_rows, schema=schema)
    result_df = test_df.join(
        prediction_df, ["demand_month", *SERIES_KEY_COLUMNS], "left"
    )
    if result_df.filter(F.col(PREDICTION_COLUMN).isNull()).limit(1).count():
        raise ValueError("Two-stage evaluation did not cover the entire test set")
    return result_df


def evaluate_two_stage_candidates(
    train_df: DataFrame,
    test_df: DataFrame,
    history_states: Mapping[SeriesKey, SeriesState],
    target_months: Sequence[date],
    thresholds: Sequence[float] = THRESHOLDS,
) -> tuple[dict[str, DataFrame], list[dict[str, Any]]]:
    """Fit once and compare expected values and thresholds on the same holdout.

    Expected values and thresholds are experimental hypotheses. Do not select
    either for production based only on this holdout; confirm using another
    temporal cutoff first.
    """
    if not thresholds or len(set(thresholds)) != len(thresholds):
        raise ValueError("Provide distinct thresholds for evaluation")
    if not history_states or not target_months:
        raise ValueError("Two-stage backtesting requires history and target months")
    for threshold in thresholds:
        gated_quantity(0.0, 0.0, threshold)
    if len({round(threshold * 100) for threshold in thresholds}) != len(thresholds):
        raise ValueError("Duplicate threshold in the model identifier")

    (
        preprocessing,
        classifier,
        magnitude,
        training_rows,
        training_seconds,
        train_occurrence_rate,
    ) = _fit_models(train_df)
    print(
        f"{MODEL_NAME} trained in {training_seconds:.2f}s "
        f"({training_rows} rows; occurrence and positive quantity)"
    )

    prediction_frames: dict[str, DataFrame] = {}
    metric_rows: list[dict[str, Any]] = []
    candidates: list[tuple[str, float | None]] = [
        (EXPECTED_MODEL_NAME, None),
        *[
            (f"{MODEL_NAME}_t{round(threshold * 100):02d}", threshold)
            for threshold in thresholds
        ],
    ]
    for model_name, threshold in candidates:
        predictions_df = _recursive_prediction_frame(
            test_df,
            history_states,
            target_months,
            preprocessing,
            classifier,
            magnitude,
            threshold,
        )
        evaluation_rows = predictions_df.select(
            LABEL_COLUMN,
            PREDICTION_COLUMN,
            "occurrence_probability",
        ).collect()
        actual = np.asarray([row[LABEL_COLUMN] for row in evaluation_rows], dtype=float)
        predicted = np.asarray(
            [row[PREDICTION_COLUMN] for row in evaluation_rows], dtype=float
        )
        metrics = calculate_metrics(actual, predicted)
        occurrence_metrics = calculate_occurrence_metrics(
            actual,
            np.asarray(
                [row["occurrence_probability"] for row in evaluation_rows],
                dtype=float,
            ),
            train_occurrence_rate,
            decision_threshold=threshold or 0.5,
        )
        prediction_frames[model_name] = predictions_df
        metric_rows.append(
            {
                "model_name": model_name,
                "model_family": MODEL_NAME,
                **metrics,
                "fallback_rows": 0,
                "parameters_json": json.dumps(
                    {
                        "forecast_rule": (
                            "expected_demand"
                            if threshold is None
                            else "threshold_gated_quantity"
                        ),
                        "occurrence_threshold": threshold,
                        "train_occurrence_rate": train_occurrence_rate,
                        "occurrence_metrics": occurrence_metrics,
                        **TREE_PARAMETERS,
                    },
                    sort_keys=True,
                ),
                "training_rows": training_rows,
                "training_seconds": training_seconds,
            }
        )
        print(f"{model_name}: WAPE={metrics['wape']:.4f}, MAE={metrics['mae']:.4f}")
    return prediction_frames, metric_rows
