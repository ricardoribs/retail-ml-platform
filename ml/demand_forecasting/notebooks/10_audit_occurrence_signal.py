# Databricks notebook source
# ruff: noqa: E402

"""Read-only audit of the sales signal by product and location.

Compares temporal windows without training models. Every predictor comes from
history before the first test month. Actual values from the three test months appear
only in the observed rates and quantities."""

from __future__ import annotations

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

from ml.demand_forecasting.src.config import (
    FORECAST_CONFIG,
    INTERMEDIATE_SCHEMA,
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
    """Displays compact results in Databricks or in the terminal."""
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(100, truncate=False)


def sale_rate_by(
    df: DataFrame, columns: tuple[str, ...], base_rate: float
) -> DataFrame:
    """Observed rate by group and lift relative to the full window."""
    grouped = df.groupBy(*columns).agg(
        F.count(F.lit(1)).alias("rows"),
        F.countDistinct(*SERIES_KEY_COLUMNS).alias("series"),
        F.sum(F.when(F.col(LABEL_COLUMN) > 0, 1).otherwise(0)).alias("sale_rows"),
        F.sum(LABEL_COLUMN).alias("actual_qty"),
    )
    rate = F.col("sale_rows") / F.col("rows")
    return grouped.withColumn("sale_rate_pct", F.round(100 * rate, 2)).withColumn(
        "lift_vs_window",
        F.when(F.lit(base_rate) > 0, F.round(rate / F.lit(base_rate), 2)),
    )


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Cutoffs and source integrity
# MAGIC
# MAGIC Run both cutoffs together. The audit reads only the feature matrix.
# MAGIC features; it does not create tables, train models, or publish forecasts.

# COMMAND ----------

dbutils.widgets.text(  # noqa: F821
    "cutoff_dates", "2025-11-30,2026-08-31"
)
raw_cutoffs = dbutils.widgets.get("cutoff_dates")  # noqa: F821
cutoff_dates = tuple(
    date.fromisoformat(value.strip())
    for value in raw_cutoffs.split(",")
    if value.strip()
)
if not cutoff_dates or len(cutoff_dates) != len(set(cutoff_dates)):
    raise ValueError("Provide distinct cutoff dates in YYYY-MM-DD format")
for cutoff in cutoff_dates:
    if cutoff != shift_months(cutoff, 1) - timedelta(days=1):
        raise ValueError(f"The cutoff must be the last day of the month: {cutoff}")
    if cutoff > FORECAST_CONFIG.cutoff_date:
        raise ValueError(f"Cutoff {cutoff} is after the last closed month")

feature_table = fq_table(INTERMEDIATE_SCHEMA, "int_forecast_feature_matrix")
feature_matrix_df = spark.table(feature_table).filter(  # noqa: F821
    F.col("demand_month") <= F.lit(max(cutoff_dates))
)
duplicate_keys = (
    feature_matrix_df.groupBy("demand_month", *SERIES_KEY_COLUMNS)
    .count()
    .filter(F.col("count") != 1)
    .count()
)
if duplicate_keys:
    raise ValueError(
            f"The matrix contains {duplicate_keys} duplicate month-product-location keys"
    )
invalid_demand_rows = feature_matrix_df.filter(
    F.col(LABEL_COLUMN).isNull() | (F.col(LABEL_COLUMN) < 0)
).count()
if invalid_demand_rows:
    raise ValueError(
        f"The matrix contains {invalid_demand_rows} null or negative demand values"
    )
print(
    f"Source: {feature_table}; duplicate keys: {duplicate_keys}; "
    f"invalid demand values: {invalid_demand_rows}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Observed recurrence without looking ahead
# MAGIC
# MAGIC For each window, recent counts are frozen at the end of training. The
# MAGIC corresponding month from the previous year was also known at the cutoff.
# MAGIC No lag from test months 2 and 3 is used as a predictor, because those
# MAGIC lags would contain actuals from months that are in the forecast horizon.

# COMMAND ----------

for cutoff in cutoff_dates:
    split = build_temporal_split(cutoff, test_months=3)
    eligible_series_df, _, test_df = prepare_temporal_frames(
        feature_matrix_df,
        split,
        FORECAST_CONFIG.min_history_months,
        FORECAST_CONFIG.min_positive_demand_months,
    )
    history_df = feature_matrix_df.filter(
        F.col("demand_month") <= F.lit(split.train_end)
    ).join(
        eligible_series_df.select(*SERIES_KEY_COLUMNS).distinct(),
        list(SERIES_KEY_COLUMNS),
        "inner",
    )
    recent_3m_start = shift_months(split.test_start, -3)
    recent_6m_start = shift_months(split.test_start, -6)
    history_signals_df = history_df.groupBy(*SERIES_KEY_COLUMNS).agg(
        F.sum(
            F.when(
                (F.col("demand_month") >= F.lit(recent_3m_start))
                & (F.col(LABEL_COLUMN) > 0),
                1,
            ).otherwise(0)
        ).alias("sales_last_3m"),
        F.sum(
            F.when(
                (F.col("demand_month") >= F.lit(recent_6m_start))
                & (F.col(LABEL_COLUMN) > 0),
                1,
            ).otherwise(0)
        ).alias("sales_last_6m"),
        F.sum(
            F.when(F.col("demand_month") >= F.lit(recent_3m_start), 1).otherwise(0)
        ).alias("observed_months_last_3m"),
        F.sum(
            F.when(F.col("demand_month") >= F.lit(recent_6m_start), 1).otherwise(0)
        ).alias("observed_months_last_6m"),
        F.sum(
            F.when(
                F.col("demand_month") >= F.lit(recent_3m_start),
                F.col(LABEL_COLUMN),
            ).otherwise(0)
        ).alias("qty_last_3m"),
        F.sum(
            F.when(
                F.col("demand_month") >= F.lit(recent_6m_start),
                F.col(LABEL_COLUMN),
            ).otherwise(0)
        ).alias("qty_last_6m"),
        F.max(F.when(F.col(LABEL_COLUMN) > 0, F.col("demand_month"))).alias(
            "last_sale_month"
        ),
    )
    prior_year_df = history_df.select(
        *SERIES_KEY_COLUMNS,
        F.col("demand_month").alias("prior_year_month"),
        F.col(LABEL_COLUMN).alias("prior_year_qty"),
    )
    audited_df = (
        test_df.join(history_signals_df, list(SERIES_KEY_COLUMNS), "left")
        .withColumn("prior_year_month", F.add_months("demand_month", -12))
        .join(
            prior_year_df,
            [*SERIES_KEY_COLUMNS, "prior_year_month"],
            "left",
        )
        .withColumn(
            "prior_year_status",
            F.when(F.col("prior_year_qty").isNull(), "no_history")
            .when(F.col("prior_year_qty") > 0, "had_sales")
            .otherwise("no_sales"),
        )
        .withColumn(
            "last_sale_recency",
            F.when(F.col("last_sale_month").isNull(), "no_historical_sales")
            .when(
                F.months_between(F.lit(split.test_start), "last_sale_month") <= 1,
                "1 month",
            )
            .when(
                F.months_between(F.lit(split.test_start), "last_sale_month") <= 3,
                "2-3 months",
            )
            .otherwise("4+ months"),
        )
    )
    test_rows = test_df.count()
    if not test_rows:
        raise ValueError(f"Cutoff {cutoff} has no eligible test rows")
    audit_rows = audited_df.count()
    if audit_rows != test_rows:
        raise ValueError(
            f"The audit join changed cardinality: {test_rows} -> {audit_rows}"
        )
    base_row = audited_df.agg(
        F.avg((F.col(LABEL_COLUMN) > 0).cast("double")).alias("sale_rate"),
        F.sum(LABEL_COLUMN).alias("actual_qty"),
    ).first()
    base_rate = float(base_row["sale_rate"])
    print(
        f"Cutoff {cutoff}: training through {split.train_end}; "
        f"teste {split.test_start} a {split.test_end}; "
        f"{test_rows} rows; sales rate {base_rate:.2%}; "
        f"actual demand {base_row['actual_qty']}"
    )

    print("Sales rate by number of months with sales in the last three")
    show_result(
        sale_rate_by(audited_df, ("sales_last_3m",), base_rate).orderBy("sales_last_3m")
    )
    print("Rate by forecast month and sales in the last three months")
    show_result(
        sale_rate_by(audited_df, ("demand_month", "sales_last_3m"), base_rate).orderBy(
            "demand_month", "sales_last_3m"
        )
    )
    print("Sales rate by number of months with sales in the last six")
    show_result(
        sale_rate_by(audited_df, ("sales_last_6m",), base_rate).orderBy("sales_last_6m")
    )
    print("Sales rate by recency of the last sale")
    show_result(
        sale_rate_by(audited_df, ("last_sale_recency",), base_rate).orderBy(
            "last_sale_recency"
        )
    )
    print("Sales rate for the same month in the previous year")
    show_result(
        sale_rate_by(audited_df, ("prior_year_status",), base_rate).orderBy(
            "prior_year_status"
        )
    )
    print("Product/category and location: highest actual volumes in the test set")
    show_result(
        sale_rate_by(
            audited_df, ("category_name", "location_name", "location_type"), base_rate
        )
        .orderBy(F.desc("actual_qty"))
        .limit(30)
    )

    # Check the attributes available in the first forecast month. Lag/rolling
    # values from test months 2 and 3 are not used in this audit.
    first_month_df = audited_df.filter(F.col("demand_month") == F.lit(split.test_start))
    for lag in (1, 2, 3):
        expected_df = history_df.filter(
            F.col("demand_month") == F.lit(shift_months(split.test_start, -lag))
        ).select(
            *SERIES_KEY_COLUMNS,
            F.col(LABEL_COLUMN).alias(f"expected_lag_{lag}_qty"),
        )
        first_month_df = first_month_df.join(
            expected_df, list(SERIES_KEY_COLUMNS), "left"
        )
    quality_df = first_month_df.agg(
        F.count(F.lit(1)).alias("series"),
        F.sum(F.when(F.col("observed_months_last_3m") != 3, 1).otherwise(0)).alias(
            "incomplete_recent_history"
        ),
        F.sum(F.when(F.col("observed_months_last_6m") != 6, 1).otherwise(0)).alias(
            "incomplete_six_month_history"
        ),
        *(
            F.sum(
                F.when(
                    ~F.col(f"lag_{lag}_qty").eqNullSafe(
                        F.col(f"expected_lag_{lag}_qty")
                    ),
                    1,
                ).otherwise(0)
            ).alias(f"lag_{lag}_mismatches")
            for lag in (1, 2, 3)
        ),
        F.sum(F.when(F.col("rolling_3m_avg_qty").isNull(), 1).otherwise(0)).alias(
            "missing_rolling_3m"
        ),
        F.sum(F.when(F.col("rolling_6m_avg_qty").isNull(), 1).otherwise(0)).alias(
            "missing_rolling_6m"
        ),
        F.sum(
            F.when(
                (F.col("observed_months_last_3m") == 3)
                & (
                    F.col("rolling_3m_avg_qty").isNull()
                    | (
                        F.abs(F.col("rolling_3m_avg_qty") - F.col("qty_last_3m") / 3)
                        > 1e-6
                    )
                ),
                1,
            ).otherwise(0)
        ).alias("rolling_3m_mismatches"),
        F.sum(
            F.when(
                (F.col("observed_months_last_6m") == 6)
                & (
                    F.col("rolling_6m_avg_qty").isNull()
                    | (
                        F.abs(F.col("rolling_6m_avg_qty") - F.col("qty_last_6m") / 6)
                        > 1e-6
                    )
                ),
                1,
            ).otherwise(0)
        ).alias("rolling_6m_mismatches"),
    )
    print("Lag integrity in the first test month")
    show_result(quality_df)
    quality = quality_df.first()
    issue_columns = [
        name for name in quality.asDict() if name != "series" and quality[name] != 0
    ]
    if issue_columns:
        print(
            "WARNING: check these columns before interpreting the signal: "
            f"{issue_columns}"
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Interpretation
# MAGIC
# MAGIC If any `*_mismatches` value or incomplete history is nonzero,
# MAGIC verify the feature matrix before changing the model. If the recent-history groups
# MAGIC recent-history and seasonal signals have `lift_vs_window` near 1 and a rate
# MAGIC For similar sales rates, these attributes provide little help in locating
# MAGIC sales at the product-location grain. If groups separate consistently,
# MAGIC review the feature representation and training strategy. This audit does not
# MAGIC select a model or publish a forecast.
