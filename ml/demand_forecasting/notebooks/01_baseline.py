# Databricks notebook source

# COMMAND ----------

"""Moving-average baseline for the demand forecast.

Reads `int_forecast_feature_matrix`, selects the test window (the latest closed months
before the cutoff), and evaluates MA3 and MA6 using the official metrics. Evaluation
logic lives in `ml.demand_forecasting.src.baseline`; this notebook reads data, builds
the window, and presents the evidence."""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any

import numpy as np
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ml.demand_forecasting.src.baseline import (
    BASELINE_SOURCE_COLUMNS,
    evaluate_baselines,
    resolve_baseline_predictions,
)
from ml.demand_forecasting.src.config import (
    FORECAST_CONFIG,
    INTERMEDIATE_SCHEMA,
    fq_table,
)

if TYPE_CHECKING:
    display: Any
    spark: Any

# COMMAND ----------


def show_result(df: DataFrame, rows: int = 100) -> None:
    """Display a DataFrame in Databricks, with a local execution fallback."""
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(rows, truncate=False)


def shift_months(reference: date, months: int) -> date:
    """Shift a reference date by months and return the first day of the resulting month."""
    month_index = reference.month - 1 + months
    year = reference.year + month_index // 12
    month = month_index % 12 + 1
    return date(year, month, 1)


FEATURE_MATRIX_TABLE = fq_table(INTERMEDIATE_SCHEMA, "int_forecast_feature_matrix")
GRAIN_COLUMNS = FORECAST_CONFIG.demand_grain
SOURCE_COLUMNS = list(BASELINE_SOURCE_COLUMNS.values())

TEST_WINDOW_MONTHS = len(FORECAST_CONFIG.horizon_months)
CUTOFF_MONTH_START = FORECAST_CONFIG.cutoff_date.replace(day=1)
TEST_WINDOW_START = shift_months(CUTOFF_MONTH_START, -(TEST_WINDOW_MONTHS - 1))
TEST_WINDOW_END = FORECAST_CONFIG.cutoff_date

print("Baseline parameters")
print(f"  Feature table      : {FEATURE_MATRIX_TABLE}")
print(f"  Cutoff date        : {FORECAST_CONFIG.cutoff_date}")
print(f"  Horizon (months)   : {TEST_WINDOW_MONTHS}")
print(f"  Test window        : {TEST_WINDOW_START} to {TEST_WINDOW_END}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Recorte da janela de teste
# MAGIC
# MAGIC Read the feature matrix and keep only months in the test window (the latest
# MAGIC closed months before the cutoff date, as described in section 6 of the
# MAGIC README). The `rolling_3m_avg_qty` and `rolling_6m_avg_qty` columns already
# MAGIC contain leakage-free moving-average forecasts.

# COMMAND ----------

feature_matrix_df = spark.table(FEATURE_MATRIX_TABLE)  # noqa: F821

test_window_df = (
    feature_matrix_df.filter(
        (F.col("demand_month") >= F.lit(TEST_WINDOW_START))
        & (F.col("demand_month") <= F.lit(TEST_WINDOW_END))
    )
    .select(*GRAIN_COLUMNS, "demand_qty", *SOURCE_COLUMNS)
    .orderBy(*GRAIN_COLUMNS)
)

print(f"Rows in the test window: {test_window_df.count()}")
show_result(test_window_df.select("demand_month").distinct().orderBy("demand_month"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Extracting the vectors
# MAGIC
# MAGIC Collect actual demand and both moving averages as NumPy arrays. Averages
# MAGIC are null for short series (the feature matrix sets them to null when history
# MAGIC is shorter than the window). The module's fallback cascade (MA6 → MA3 →
# MAGIC zero, and MA3 → zero) resolves these nulls for evaluation and the forecast
# MAGIC table.

# COMMAND ----------

test_rows = test_window_df.collect()

ma3_column = BASELINE_SOURCE_COLUMNS["baseline_ma3"]
ma6_column = BASELINE_SOURCE_COLUMNS["baseline_ma6"]

actual = np.array([row["demand_qty"] for row in test_rows], dtype=float)
ma3_predicted = np.array([row[ma3_column] for row in test_rows], dtype=float)
ma6_predicted = np.array([row[ma6_column] for row in test_rows], dtype=float)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Forecast baseline
# MAGIC
# MAGIC Build the baseline forecast table for the test window in long format (one
# MAGIC row per series and model). `predicted_qty` contains the effective forecast
# MAGIC after the fallback cascade (MA6 → MA3 → zero, and MA3 → zero), so it has no
# MAGIC nulls, including for short series.

# COMMAND ----------

effective_predictions = resolve_baseline_predictions(ma3_predicted, ma6_predicted)

forecast_rows = []
for index, row in enumerate(test_rows):
    grain_values = {column: row[column] for column in GRAIN_COLUMNS}
    for model_name in BASELINE_SOURCE_COLUMNS:
        forecast_rows.append(
            {
                **grain_values,
                "demand_qty": row["demand_qty"],
                "predicted_qty": float(effective_predictions[model_name][index]),
                "model_name": model_name,
            }
        )

baseline_forecast_df = spark.createDataFrame(forecast_rows).orderBy(  # noqa: F821
    *GRAIN_COLUMNS, "model_name"
)

show_result(baseline_forecast_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Baseline metrics
# MAGIC
# MAGIC Evaluate MA3 and MA6 using the official metrics (MAE, RMSE, MAPE, WAPE) on
# MAGIC all rows after applying the fallback cascade. The `fallback_rows` field
# MAGIC indicates how many observations used the fallback (short series), now
# MAGIC included in the evaluation instead of being discarded.

# COMMAND ----------

baseline_metrics = evaluate_baselines(actual, ma3_predicted, ma6_predicted)

metrics_df = spark.createDataFrame(list(baseline_metrics.values()))  # noqa: F821
show_result(metrics_df)

for model_name, metrics in baseline_metrics.items():
    print(f"\n=== {model_name} ===")
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name}: {metric_value}")

# COMMAND ----------

print("MOVING AVERAGE BASELINE COMPLETED")
