"""Persistence for forecast model metrics in Unity Catalog."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pyspark.sql import DataFrame, SparkSession


METRICS_TABLE_NAME = "fct_forecast_model_metrics"


@dataclass(frozen=True)
class MetricRunMetadata:
    """Metadata shared by all models in a run."""

    run_id: str
    model_version: str
    train_start_month: date
    train_end_month: date
    test_start_month: date
    test_end_month: date
    cutoff_date: date

    @property
    def evaluation_window(self) -> str:
        """Stable identifier for the evaluation time window."""
        return f"{self.test_start_month.isoformat()}/{self.test_end_month.isoformat()}"


def _nullable_metric(value: Any) -> float | None:
    """Convert undefined metrics (NaN/inf) to NULL for persistence."""
    numeric_value = float(value)
    return numeric_value if math.isfinite(numeric_value) else None


def _required_metric(value: Any, metric_name: str) -> float:
    """Validate required metrics before creating the Spark DataFrame."""
    numeric_value = float(value)
    if not math.isfinite(numeric_value):
        raise ValueError(f"{metric_name} must be a finite metric")
    return numeric_value


def prepare_metric_records(
    metric_rows: list[dict[str, Any]],
    metadata: MetricRunMetadata,
    best_model_name: str,
    created_at: datetime | None = None,
) -> list[dict[str, Any]]:
    """Validate and enrich metrics for the final table contract."""
    if not metric_rows:
        raise ValueError("metric_rows cannot be empty")
    if not metadata.run_id.strip():
        raise ValueError("run_id cannot be empty")
    if not metadata.model_version.strip():
        raise ValueError("model_version cannot be empty")

    model_names = [str(row.get("model_name", "")).strip() for row in metric_rows]
    if any(not model_name for model_name in model_names):
        raise ValueError("every metric must include model_name")
    if len(model_names) != len(set(model_names)):
        raise ValueError("model_name must be unique within a run")
    if best_model_name not in model_names:
        raise ValueError("best_model_name must exist in metric_rows")

    persisted_at = created_at or datetime.now(timezone.utc).replace(tzinfo=None)
    records: list[dict[str, Any]] = []

    for row, model_name in zip(metric_rows, model_names, strict=True):
        evaluated_rows = int(row["evaluated_rows"])
        if evaluated_rows <= 0:
            raise ValueError("evaluated_rows must be greater than zero")

        records.append(
            {
                "run_id": metadata.run_id,
                "model_name": model_name,
                "model_family": str(row.get("model_family", "unknown")),
                "model_version": metadata.model_version,
                "evaluation_window": metadata.evaluation_window,
                "train_start_month": metadata.train_start_month,
                "train_end_month": metadata.train_end_month,
                "test_start_month": metadata.test_start_month,
                "test_end_month": metadata.test_end_month,
                "cutoff_date": metadata.cutoff_date,
                "mae": _required_metric(row["mae"], "mae"),
                "rmse": _required_metric(row["rmse"], "rmse"),
                "mape": _nullable_metric(row["mape"]),
                "wape": _nullable_metric(row["wape"]),
                "evaluated_rows": evaluated_rows,
                "fallback_rows": int(row.get("fallback_rows", 0)),
                "training_rows": int(row.get("training_rows", 0)),
                "training_seconds": float(row.get("training_seconds", 0.0)),
                "parameters_json": str(row.get("parameters_json", "{}")),
                "is_best_model": model_name == best_model_name,
                "created_at": persisted_at,
            }
        )

    return records


def _metrics_schema() -> Any:
    """Return the explicit Spark schema for the metrics table."""
    from pyspark.sql.types import (
        BooleanType,
        DateType,
        DoubleType,
        LongType,
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    return StructType(
        [
            StructField("run_id", StringType(), False),
            StructField("model_name", StringType(), False),
            StructField("model_family", StringType(), False),
            StructField("model_version", StringType(), False),
            StructField("evaluation_window", StringType(), False),
            StructField("train_start_month", DateType(), False),
            StructField("train_end_month", DateType(), False),
            StructField("test_start_month", DateType(), False),
            StructField("test_end_month", DateType(), False),
            StructField("cutoff_date", DateType(), False),
            StructField("mae", DoubleType(), False),
            StructField("rmse", DoubleType(), False),
            StructField("mape", DoubleType(), True),
            StructField("wape", DoubleType(), True),
            StructField("evaluated_rows", LongType(), False),
            StructField("fallback_rows", LongType(), False),
            StructField("training_rows", LongType(), False),
            StructField("training_seconds", DoubleType(), False),
            StructField("parameters_json", StringType(), False),
            StructField("is_best_model", BooleanType(), False),
            StructField("created_at", TimestampType(), False),
        ]
    )


def persist_model_metrics(
    spark: SparkSession,
    records: list[dict[str, Any]],
    table_name: str,
) -> DataFrame:
    """Persist metrics to Delta with an idempotent upsert by model and window."""
    if not records:
        raise ValueError("records cannot be empty")

    metrics_df = spark.createDataFrame(records, schema=_metrics_schema())

    if spark.catalog.tableExists(table_name):
        from delta.tables import DeltaTable

        target = DeltaTable.forName(spark, table_name)
        condition = (
            "target.run_id = source.run_id "
            "AND target.model_name = source.model_name "
            "AND target.evaluation_window = source.evaluation_window"
        )
        (
            target.alias("target")
            .merge(metrics_df.alias("source"), condition)
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .whenNotMatchedBySourceDelete(
                condition=f"target.run_id = '{records[0]['run_id']}'"
            )
            .execute()
        )
    else:
        # Append mode creates the table when it does not yet exist and is supported
        # by serverless compute. Subsequent runs use the MERGE above.
        metrics_df.write.format("delta").mode("append").saveAsTable(table_name)

    from pyspark.sql import functions as F

    run_id = records[0]["run_id"]
    return spark.table(table_name).where(F.col("run_id") == F.lit(run_id))
