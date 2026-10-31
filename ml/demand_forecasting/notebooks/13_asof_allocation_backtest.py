# Databricks notebook source
# ruff: noqa: E402

"""Audits the target and compares as-of seasonal allocation at two historical cutoffs.

Read-only: it does not write tables, update models, or change the job. Candidates
are fixed rules with no hyperparameters selected from the holdouts."""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyspark.sql import DataFrame, Window
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

from ml.demand_forecasting.src.asof_allocation import (
    ADAPTIVE_METHODS,
    SERIES_BASELINES,
    build_asof_forecasts,
    history_segment,
)
from ml.demand_forecasting.src.config import (
    FORECAST_CONFIG,
    INTERMEDIATE_SCHEMA,
    MARTS_SCHEMA,
    fq_table,
)
from ml.demand_forecasting.src.hierarchical import (
    FORECAST_METHODS,
    evaluate_forecasts,
)
from ml.demand_forecasting.src.modeling import (
    LABEL_COLUMN,
    SERIES_KEY_COLUMNS,
    build_temporal_split,
    prepare_temporal_frames,
    shift_months,
)


def show_result(df: DataFrame) -> None:
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(200, truncate=False)


def finite_round(value: float, digits: int = 2) -> float | None:
    return round(value, digits) if math.isfinite(value) else None


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Cutoffs and target definition
# MAGIC
# MAGIC The current target is **quantity sold in paid orders**. Rows set to zero by
# MAGIC the calendar mean *no paid sale was recorded*, not that demand was absent.
# MAGIC Without reliable assortment and inventory history, these zeros are not
# MAGIC recoded as lost demand. Notebook 12 describes the limits of inventory
# MAGIC reconstruction. Here, we verify that the target exactly matches `fct_sales`
# MAGIC for the evaluated cohort.

# COMMAND ----------

dbutils.widgets.text("cutoff_dates", "2025-11-30,2026-08-31")  # noqa: F821
cutoff_dates = tuple(
    date.fromisoformat(value.strip())
    for value in dbutils.widgets.get("cutoff_dates").split(",")  # noqa: F821
    if value.strip()
)
if not cutoff_dates or len(cutoff_dates) != len(set(cutoff_dates)):
    raise ValueError("Informe cortes distintos no formato YYYY-MM-DD")
for cutoff in cutoff_dates:
    if cutoff != shift_months(cutoff, 1) - timedelta(days=1):
        raise ValueError(f"The cutoff must be the last day of the month: {cutoff}")
    if cutoff > FORECAST_CONFIG.cutoff_date:
        raise ValueError(f"Cutoff is after the last closed month: {cutoff}")

feature_matrix_df = spark.table(  # noqa: F821
    fq_table(INTERMEDIATE_SCHEMA, "int_forecast_feature_matrix")
).filter(F.col("demand_month") <= F.lit(max(cutoff_dates)))
duplicate_keys = (
    feature_matrix_df.groupBy("demand_month", *SERIES_KEY_COLUMNS)
    .count()
    .filter(F.col("count") != 1)
    .count()
)
invalid_labels = feature_matrix_df.filter(
    F.col(LABEL_COLUMN).isNull() | (F.col(LABEL_COLUMN) < 0)
).count()
if duplicate_keys or invalid_labels:
    raise ValueError(
        f"Invalid matrix: {duplicate_keys} duplicate keys, "
        f"{invalid_labels} quantidades nulas/negativas"
    )

sales_df = spark.table(fq_table(MARTS_SCHEMA, "fct_sales"))  # noqa: F821
locations_df = spark.table(fq_table(MARTS_SCHEMA, "dim_location"))  # noqa: F821
first_test_month = min(
    build_temporal_split(cutoff, 3).test_start for cutoff in cutoff_dates
)
paid_sales_by_month_df = (
    sales_df.filter(F.col("order_status") == "paid")
    .filter(F.to_date("sale_datetime") >= F.lit(first_test_month))
    .filter(F.to_date("sale_datetime") <= F.lit(max(cutoff_dates)))
    .join(locations_df.select("location_key", "location_id"), "location_key")
    .withColumn("demand_month", F.trunc(F.to_date("sale_datetime"), "month"))
    .groupBy("demand_month", *SERIES_KEY_COLUMNS)
    .agg(F.sum("quantity_qty").alias("paid_sales_qty"))
)

target_audit_rows: list[dict[str, object]] = []
score_rows: list[dict[str, object]] = []
monthly_rows: list[dict[str, object]] = []

for cutoff in cutoff_dates:
    split = build_temporal_split(cutoff, test_months=3)
    target_months = tuple(shift_months(split.test_start, i) for i in range(3))
    eligible_df, _, test_df = prepare_temporal_frames(
        feature_matrix_df,
        split,
        FORECAST_CONFIG.min_history_months,
        FORECAST_CONFIG.min_positive_demand_months,
    )
    eligible_count = eligible_df.count()
    full_test_df = feature_matrix_df.filter(
        (F.col("demand_month") >= F.lit(split.test_start))
        & (F.col("demand_month") <= F.lit(split.test_end))
    )
    full_summary = full_test_df.agg(
        F.countDistinct(*SERIES_KEY_COLUMNS).alias("all_series"),
        F.sum(LABEL_COLUMN).alias("all_qty"),
    ).first()
    if not eligible_count or full_summary["all_series"] < eligible_count:
        raise ValueError(f"Invalid cohort at cutoff {cutoff}")

    source_audit_df = test_df.join(
        paid_sales_by_month_df,
        ["demand_month", *SERIES_KEY_COLUMNS],
        "left",
    )
    source_audit = source_audit_df.agg(
        F.count(F.lit(1)).alias("rows"),
        F.sum(F.when(F.col(LABEL_COLUMN) == 0, 1).otherwise(0)).alias("zero_rows"),
        F.sum(F.when(F.col("paid_sales_qty").isNull(), 1).otherwise(0)).alias(
            "months_without_paid_sale"
        ),
        F.sum(
            F.when(
                F.abs(
                    F.col(LABEL_COLUMN) - F.coalesce(F.col("paid_sales_qty"), F.lit(0))
                )
                > 1e-6,
                1,
            ).otherwise(0)
        ).alias("source_mismatches"),
        F.sum(LABEL_COLUMN).alias("eligible_qty"),
    ).first()
    if source_audit["source_mismatches"]:
        raise ValueError(
            f"Target does not reconcile with paid sales for {cutoff}: "
            f"{source_audit['source_mismatches']} linhas"
        )
    target_audit_rows.append(
        {
            "cutoff": cutoff.isoformat(),
            "all_grid_series": int(full_summary["all_series"]),
            "eligible_series": eligible_count,
            "all_grid_qty": float(full_summary["all_qty"]),
            "eligible_qty": float(source_audit["eligible_qty"]),
            "eligible_rows": int(source_audit["rows"]),
            "zero_sale_rows": int(source_audit["zero_rows"]),
            "months_without_paid_sale": int(source_audit["months_without_paid_sale"]),
            "source_mismatches": int(source_audit["source_mismatches"]),
        }
    )

    # Category and location type are fixed using the last pre-test record.
    history_df = feature_matrix_df.filter(
        (F.col("demand_month") >= F.lit(shift_months(split.test_start, -24)))
        & (F.col("demand_month") <= F.lit(split.train_end))
    ).join(eligible_df.select(*SERIES_KEY_COLUMNS), list(SERIES_KEY_COLUMNS))
    latest = Window.partitionBy(*SERIES_KEY_COLUMNS).orderBy(
        F.col("demand_month").desc()
    )
    metadata_df = (
        history_df.withColumn("_rank", F.row_number().over(latest))
        .filter(F.col("_rank") == 1)
        .select(
            *SERIES_KEY_COLUMNS,
            F.coalesce(F.col("category_name"), F.lit("__missing__")).alias("_category"),
            F.coalesce(F.col("location_type"), F.lit("__missing__")).alias(
                "_location_type"
            ),
        )
    )
    metadata_rows = metadata_df.collect()
    series_groups = {
        (row["product_variant_id"], row["location_id"]): (
            row["location_id"],
            row["_category"],
        )
        for row in metadata_rows
    }
    series_types = {
        (row["product_variant_id"], row["location_id"]): row["_location_type"]
        for row in metadata_rows
    }
    if len(metadata_rows) != eligible_count or len(series_groups) != eligible_count:
        raise ValueError(f"Metadata is incomplete or duplicated at {cutoff}")

    series_history = {series: {} for series in series_groups}
    for row in history_df.select(
        "demand_month", *SERIES_KEY_COLUMNS, LABEL_COLUMN
    ).collect():
        series = (row["product_variant_id"], row["location_id"])
        month = row["demand_month"]
        if month in series_history[series]:
            raise ValueError(f"Duplicate history for {cutoff}: {series}, {month}")
        series_history[series][month] = float(row[LABEL_COLUMN])
    segments = {
        series: history_segment(history, split.test_start)
        for series, history in series_history.items()
    }

    group_month_df = (
        history_df.drop("category_name")
        .join(metadata_df, list(SERIES_KEY_COLUMNS))
        .groupBy("demand_month", "location_id", "_category")
        .agg(F.sum(LABEL_COLUMN).alias("quantity"))
    )
    group_history = defaultdict(dict)
    for row in group_month_df.collect():
        group_history[(row["location_id"], row["_category"])][row["demand_month"]] = (
            float(row["quantity"])
        )

    actual_rows = test_df.select(
        "demand_month", *SERIES_KEY_COLUMNS, LABEL_COLUMN
    ).collect()
    actual = {
        (row["demand_month"], row["product_variant_id"], row["location_id"]): float(
            row[LABEL_COLUMN]
        )
        for row in actual_rows
    }
    expected_keys = {
        (month, product_id, location_id)
        for month in target_months
        for product_id, location_id in series_groups
    }
    if len(actual_rows) != len(actual) or set(actual) != expected_keys:
        raise ValueError(f"Incomplete or duplicate holdout for {cutoff}")

    forecasts = build_asof_forecasts(
        series_groups=series_groups,
        series_history=series_history,
        group_history=group_history,
        target_months=target_months,
    )
    scopes = {"all": set(series_groups)}
    store_series = {
        series
        for series, location_type in series_types.items()
        if location_type == "store"
    }
    scopes["store_all"] = store_series
    for segment in sorted(set(segments.values())):
        scopes[f"store_{segment}"] = {
            series for series in store_series if segments[series] == segment
        }

    for method, predicted in forecasts.items():
        for scope, selected_series in scopes.items():
            if not selected_series:
                continue
            selected_actual = {
                key: qty for key, qty in actual.items() if key[1:] in selected_series
            }
            selected_predicted = {key: predicted[key] for key in selected_actual}
            selected_groups = {
                series: series_groups[series] for series in selected_series
            }
            metrics = evaluate_forecasts(
                selected_actual, selected_predicted, selected_groups
            )
            actual_qty = sum(selected_actual.values())
            predicted_qty = sum(selected_predicted.values())
            score_rows.append(
                {
                    "cutoff": cutoff.isoformat(),
                    "scope": scope,
                    "method": method,
                    "series": len(selected_series),
                    "series_coverage_pct": round(
                        100 * len(selected_series) / int(full_summary["all_series"]), 2
                    ),
                    "actual_share_pct": finite_round(
                        100 * actual_qty / float(full_summary["all_qty"])
                        if full_summary["all_qty"]
                        else math.nan
                    ),
                    "actual_qty": round(actual_qty, 2),
                    "predicted_qty": round(predicted_qty, 2),
                    "bias_pct": finite_round(
                        100 * (predicted_qty - actual_qty) / actual_qty
                        if actual_qty
                        else math.nan
                    ),
                    "sku_location_wape": finite_round(metrics["sku_location"]["wape"]),
                    "sku_location_mae": finite_round(metrics["sku_location"]["mae"], 3),
                    "location_wape": finite_round(metrics["location"]["wape"]),
                }
            )
        for month in target_months:
            for scope, selected_series in (
                ("all", scopes["all"]),
                ("store_all", store_series),
            ):
                if not selected_series:
                    continue
                month_actual = {
                    key: qty
                    for key, qty in actual.items()
                    if key[0] == month and key[1:] in selected_series
                }
                month_predicted = {key: predicted[key] for key in month_actual}
                month_groups = {
                    series: series_groups[series] for series in selected_series
                }
                metrics = evaluate_forecasts(
                    month_actual, month_predicted, month_groups
                )["sku_location"]
                monthly_rows.append(
                    {
                        "cutoff": cutoff.isoformat(),
                        "month": month.isoformat(),
                        "scope": scope,
                        "method": method,
                        "actual_qty": round(sum(month_actual.values()), 2),
                        "predicted_qty": round(sum(month_predicted.values()), 2),
                        "sku_location_wape": finite_round(metrics["wape"]),
                    }
                )
    print(
        f"{cutoff}: {eligible_count}/{full_summary['all_series']} eligible series, "
        f"{source_audit['zero_rows']}/{source_audit['rows']} months with no sales; "
        "no holdout data was used in the forecast"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Results and interpretation guardrail
# MAGIC
# MAGIC `store_all` is the requirement's primary grain. Segments are defined
# MAGIC using the **six months before** the holdout, and coverage is shown alongside
# MAGIC alongside the error. Improving only `location_wape`, one cutoff, or the
# MAGIC recurring series does not validate forecasts for every product and store.
# MAGIC This notebook does not automatically select or publish a model.

# COMMAND ----------

show_result(spark.createDataFrame(target_audit_rows).orderBy("cutoff"))  # noqa: F821
show_result(  # noqa: F821
    spark.createDataFrame(score_rows).orderBy("cutoff", "scope", "method")  # noqa: F821
)
show_result(  # noqa: F821
    spark.createDataFrame(monthly_rows).orderBy(  # noqa: F821
        "cutoff", "scope", "month", "method"
    )
)

print("Exploratory floor for adaptive candidates at the product/store grain:")
for method in ADAPTIVE_METHODS:
    checks = []
    for cutoff in cutoff_dates:
        cutoff_name = cutoff.isoformat()
        store_scores = {
            row["method"]: row["sku_location_wape"]
            for row in score_rows
            if row["cutoff"] == cutoff_name and row["scope"] == "store_all"
        }
        references = tuple(
            store_scores[name] for name in (*SERIES_BASELINES, *FORECAST_METHODS)
        )
        candidate = store_scores[method]
        monthly_candidate = [
            row["sku_location_wape"]
            for row in monthly_rows
            if row["cutoff"] == cutoff_name
            and row["scope"] == "store_all"
            and row["method"] == method
        ]
        passed = (
            candidate is not None
            and all(reference is not None for reference in references)
            and candidate < min(references)
            and len(monthly_candidate) == 3
            and all(value is not None and value < 100 for value in monthly_candidate)
        )
        checks.append(passed)
        print(
            f"  {method}, {cutoff_name}: WAPE={candidate}, "
            f"best benchmark={min(references) if None not in references else None}, "
            f"passou piso={passed}"
        )
    print(f"  {method}: passou os dois cortes={len(checks) >= 2 and all(checks)}")

print(
    "Exceeding the floor does not mean production approval: the metric measures "
    "observed sales, not lost demand. Validate assortment/inventory "
    "history and repeat the analysis across independent periods before changing the job."
)
