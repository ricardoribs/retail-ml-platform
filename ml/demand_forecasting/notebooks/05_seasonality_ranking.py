# Databricks notebook source
# ruff: noqa: E402

# COMMAND ----------

"""Ranks products by seasonal evidence.

Reads `int_monthly_product_location_demand` through the cutoff, calculates product
eligibility (aggregating across locations), the seasonal index and coefficient of
variation (CV) by calendar month, and ranks eligible products by descending CV.
Threshold and classification logic lives in `ml.demand_forecasting.src.seasonality`;
this notebook reads and aggregates data in Spark, presents evidence, and persists
monthly seasonality to `agg_product_seasonality` (key:
`product_variant_id + month_number`)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

if TYPE_CHECKING:
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
from ml.demand_forecasting.src.seasonality import (
    ANALYSIS_LOCATION_TYPES,
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    HIGH_CONFIDENCE_MIN_OBSERVATIONS,
    LOCATION_TYPE_STORE,
    LOCATION_TYPE_WAREHOUSE,
    MEDIUM_CONFIDENCE_MIN_OBSERVATIONS,
    MONTHS_IN_YEAR,
    SEASONAL_LOW_INDEX_THRESHOLD,
    SEASONAL_PEAK_INDEX_THRESHOLD,
    SEASONALITY_CLASS_LOW,
    SEASONALITY_CLASS_NEUTRAL,
    SEASONALITY_CLASS_PEAK,
    SEASONALITY_MIN_HISTORY_MONTHS,
    SEASONALITY_MIN_POSITIVE_MONTHS,
)

# COMMAND ----------


def show_result(df: DataFrame, rows: int = 100) -> None:
    """Display a DataFrame in Databricks, with a local execution fallback."""
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(rows, truncate=False)


def _product_month_demand(scoped_demand_df: DataFrame) -> DataFrame:
    """Sum demand by product and month, aggregating the provided locations.

    Output grain: product_variant_id + demand_month (plus month_number and
    year_number, which are constant within a month).
    """
    return scoped_demand_df.groupBy(
        "product_variant_id", "demand_month", "month_number", "year_number"
    ).agg(F.sum(F.col("demand_qty").cast("double")).alias("demand_qty"))


def _monthly_avg_and_observations(product_month_df: DataFrame) -> DataFrame:
    """Average by month_number (1..12) and number of observed years, by product.

    `observations` counts distinct years in which the month is part of the
    product's active window (the row exists, even when demand_qty = 0).
    """
    return product_month_df.groupBy("product_variant_id", "month_number").agg(
        F.avg("demand_qty").alias("avg_monthly_demand_qty"),
        F.countDistinct("year_number").alias("observations"),
    )


def _overall_avg_and_cv(monthly_avg_df: DataFrame) -> DataFrame:
    """overall_avg (mean of the 12 monthly averages), CV, and min_observations.

    overall_avg_monthly_demand_qty is the mean of the product's 12 monthly
    averages and is the denominator for both the index and CV. CV is the
    stddev_pop of these 12 averages divided by the same overall_avg.
    """
    return (
        monthly_avg_df.groupBy("product_variant_id")
        .agg(
            F.avg("avg_monthly_demand_qty").alias("overall_avg_monthly_demand_qty"),
            F.stddev_pop("avg_monthly_demand_qty").alias("monthly_avg_stddev_pop"),
            F.min("observations").alias("min_observations"),
        )
        .withColumn(
            "cv",
            F.when(
                F.col("overall_avg_monthly_demand_qty") > F.lit(0.0),
                F.col("monthly_avg_stddev_pop")
                / F.col("overall_avg_monthly_demand_qty"),
            ).otherwise(F.lit(None).cast("double")),
        )
    )


MONTHLY_DEMAND_TABLE = fq_table(
    INTERMEDIATE_SCHEMA,
    "int_monthly_product_location_demand",
)
PRODUCT_TABLE = fq_table(MARTS_SCHEMA, "dim_product")
SEASONALITY_TABLE = fq_table(MARTS_SCHEMA, "agg_product_seasonality")
CUTOFF_DATE = FORECAST_CONFIG.cutoff_date
LOCATION_SCOPE_ALL = "all"

print("Seasonality ranking parameters")
print(f"  Monthly demand : {MONTHLY_DEMAND_TABLE}")
print(f"  Products       : {PRODUCT_TABLE}")
print(f"  Cutoff date    : {CUTOFF_DATE}")
print(
    f"  Eligibility    : history_months >= {SEASONALITY_MIN_HISTORY_MONTHS} "
    f"and positive_months >= {SEASONALITY_MIN_POSITIVE_MONTHS}"
)
print(f"  Output         : {SEASONALITY_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Product eligibility (aggregating locations)
# MAGIC
# MAGIC Eligibility is calculated by `product_variant_id`, aggregating all
# MAGIC locations (store and warehouse). `history_months` counts distinct
# MAGIC `demand_month` values in the product series through the cutoff date;
# MAGIC `positive_months` counts months in which demand summed across locations
# MAGIC was greater than zero. Eligible products need at least 12 months of history
# MAGIC and 6 months with positive demand. Others are excluded from the ranking but
# MAGIC counted below.

# COMMAND ----------

monthly_demand_df = spark.table(MONTHLY_DEMAND_TABLE).filter(  # noqa: F821
    (F.col("demand_month") <= F.lit(CUTOFF_DATE))
    & (F.col("location_type").isin(*ANALYSIS_LOCATION_TYPES))
)

all_locations_product_month_df = _product_month_demand(monthly_demand_df)

product_eligibility_df = (
    all_locations_product_month_df.groupBy("product_variant_id")
    .agg(
        F.countDistinct("demand_month").alias("history_months"),
        F.sum(F.when(F.col("demand_qty") > F.lit(0.0), 1).otherwise(0)).alias(
            "positive_months"
        ),
    )
    .withColumn(
        "is_eligible",
        (F.col("history_months") >= F.lit(SEASONALITY_MIN_HISTORY_MONTHS))
        & (F.col("positive_months") >= F.lit(SEASONALITY_MIN_POSITIVE_MONTHS)),
    )
)

eligible_product_count = product_eligibility_df.filter(F.col("is_eligible")).count()
excluded_product_count = product_eligibility_df.filter(~F.col("is_eligible")).count()

print(f"Eligible products: {eligible_product_count}")
print(f"Products excluded by eligibility: {excluded_product_count}")
show_result(product_eligibility_df.orderBy("product_variant_id"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Monthly seasonality (product_variant_id + month_number)
# MAGIC
# MAGIC For eligible products only, `avg_monthly_demand_qty[m]` is the average
# MAGIC demand (aggregated across locations) for observed years of month `m`.
# MAGIC `overall_avg_monthly_demand_qty` is the mean of the product's 12 monthly
# MAGIC averages and is also the denominator for the index and CV (section 3).
# MAGIC `seasonality_index[m] = avg_monthly_demand_qty[m] /
# MAGIC overall_avg_monthly_demand_qty`; `seasonality_class` is `peak` when the
# MAGIC index is at least 1.20, `low` when it is at most 0.80, and `neutral` otherwise.

# COMMAND ----------

eligible_product_ids_df = product_eligibility_df.filter(F.col("is_eligible")).select(
    "product_variant_id"
)

eligible_all_locations_product_month_df = all_locations_product_month_df.join(
    eligible_product_ids_df, "product_variant_id", "inner"
)

monthly_avg_all_df = _monthly_avg_and_observations(
    eligible_all_locations_product_month_df
)
overall_avg_and_cv_all_df = _overall_avg_and_cv(monthly_avg_all_df)

seasonality_month_df = (
    monthly_avg_all_df.join(
        overall_avg_and_cv_all_df.select(
            "product_variant_id", "overall_avg_monthly_demand_qty"
        ),
        "product_variant_id",
        "inner",
    )
    .withColumn(
        "seasonality_index",
        F.when(
            F.col("overall_avg_monthly_demand_qty") > F.lit(0.0),
            F.col("avg_monthly_demand_qty") / F.col("overall_avg_monthly_demand_qty"),
        ).otherwise(F.lit(None).cast("double")),
    )
    .withColumn(
        "seasonality_class",
        F.when(F.col("seasonality_index").isNull(), F.lit(None).cast("string"))
        .when(
            F.col("seasonality_index") >= F.lit(SEASONAL_PEAK_INDEX_THRESHOLD),
            F.lit(SEASONALITY_CLASS_PEAK),
        )
        .when(
            F.col("seasonality_index") <= F.lit(SEASONAL_LOW_INDEX_THRESHOLD),
            F.lit(SEASONALITY_CLASS_LOW),
        )
        .otherwise(F.lit(SEASONALITY_CLASS_NEUTRAL)),
    )
    .select(
        "product_variant_id",
        "month_number",
        "avg_monthly_demand_qty",
        "overall_avg_monthly_demand_qty",
        "seasonality_index",
        "seasonality_class",
        "observations",
    )
)

show_result(
    seasonality_month_df.orderBy("product_variant_id", "month_number"), rows=200
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.1 Monthly grain integrity check
# MAGIC
# MAGIC Check that `seasonality_month_df` (grain `product_variant_id` +
# MAGIC `month_number`, eligible products only) has exactly `MONTHS_IN_YEAR` rows
# MAGIC for each `product_variant_id` and no duplicate `product_variant_id` +
# MAGIC `month_number` pairs. Raise `AssertionError` if either condition fails.

# COMMAND ----------

rows_per_product_df = seasonality_month_df.groupBy("product_variant_id").agg(
    F.count(F.lit(1)).alias("row_count")
)
wrong_row_count_df = rows_per_product_df.filter(
    F.col("row_count") != F.lit(MONTHS_IN_YEAR)
)

duplicated_month_pairs_df = (
    seasonality_month_df.groupBy("product_variant_id", "month_number")
    .agg(F.count(F.lit(1)).alias("pair_count"))
    .filter(F.col("pair_count") > F.lit(1))
)

wrong_row_count_total = wrong_row_count_df.count()
duplicated_month_pairs_total = duplicated_month_pairs_df.count()

if wrong_row_count_total > 0 or duplicated_month_pairs_total > 0:
    print(f"Products without {MONTHS_IN_YEAR} rows at the monthly grain:")
    show_result(wrong_row_count_df.orderBy("product_variant_id"))
    print("Duplicate product_variant_id + month_number pairs:")
    show_result(duplicated_month_pairs_df.orderBy("product_variant_id", "month_number"))
    raise AssertionError(
        "Invalid product_variant_id + month_number grain: "
        f"{wrong_row_count_total} product(s) without {MONTHS_IN_YEAR} rows and "
        f"{duplicated_month_pairs_total} duplicate pair(s)."
    )

validated_product_count = rows_per_product_df.count()
print(
    f"Monthly grain validated: {validated_product_count} eligible products, "
    f"all with {MONTHS_IN_YEAR} rows and no duplicate pairs."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2.2 Seasonality deep dive: product 608
# MAGIC
# MAGIC Display the full monthly series for the selected product to support and
# MAGIC validate the seasonality analysis.

# COMMAND ----------

DEEP_DIVE_PRODUCT_ID = 608

deep_dive_product_seasonality_df = seasonality_month_df.filter(
    F.col("product_variant_id") == F.lit(DEEP_DIVE_PRODUCT_ID)
).orderBy("month_number")

show_result(deep_dive_product_seasonality_df, rows=MONTHS_IN_YEAR)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. CV, confidence, and product ranking
# MAGIC
# MAGIC `cv` is the stddev_pop of the 12 monthly averages divided by the same
# MAGIC `overall_avg_monthly_demand_qty` from section 2. `confidence` uses the
# MAGIC lowest `observations` value across the product's 12 months (`min_obs`):
# MAGIC high when `min_obs >= 3`, medium when it is 2, and low when it is 0 or 1.
# MAGIC This flag does not exclude products from the ranking. Rank by descending
# MAGIC `cv`, using `product_variant_id` to break ties.

# COMMAND ----------

ranking_window = Window.orderBy(
    F.col("cv").desc_nulls_last(), F.col("product_variant_id")
)

product_seasonality_ranking_df = (
    product_eligibility_df.filter(F.col("is_eligible"))
    .join(
        overall_avg_and_cv_all_df.select(
            "product_variant_id", "cv", "min_observations"
        ),
        "product_variant_id",
        "inner",
    )
    .withColumn(
        "confidence",
        F.when(
            F.col("min_observations") >= F.lit(HIGH_CONFIDENCE_MIN_OBSERVATIONS),
            F.lit(CONFIDENCE_HIGH),
        )
        .when(
            F.col("min_observations") >= F.lit(MEDIUM_CONFIDENCE_MIN_OBSERVATIONS),
            F.lit(CONFIDENCE_MEDIUM),
        )
        .otherwise(F.lit(CONFIDENCE_LOW)),
    )
    .withColumn("rank", F.rank().over(ranking_window))
    .withColumn("location_scope", F.lit(LOCATION_SCOPE_ALL))
    .select(
        "product_variant_id",
        "history_months",
        "positive_months",
        "is_eligible",
        "cv",
        "confidence",
        "rank",
        "location_scope",
    )
)

ranked_product_count = product_seasonality_ranking_df.count()
print(
    f"Produtos ranqueados (location_scope = {LOCATION_SCOPE_ALL}): "
    f"{ranked_product_count}"
)
show_result(product_seasonality_ranking_df.orderBy("rank"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Additional check: store vs. warehouse vs. all
# MAGIC
# MAGIC Recalculate CV for the same products and window, separately for `store`
# MAGIC and `warehouse` locations, then compare each with the `all` CV from section
# MAGIC 3. This helps identify products whose CV changes substantially across scopes
# MAGIC (which may indicate distortion from one location type). This check is evidence
# MAGIC only and is not persisted.

# COMMAND ----------


def _cv_by_scope(scope_demand_df: DataFrame, scope_label: str) -> DataFrame:
    """Calculate product CV for the location scope already filtered in scope_demand_df."""
    scoped_eligible_df = scope_demand_df.join(
        eligible_product_ids_df, "product_variant_id", "inner"
    )
    product_month_df = _product_month_demand(scoped_eligible_df)
    monthly_avg_df = _monthly_avg_and_observations(product_month_df)
    return _overall_avg_and_cv(monthly_avg_df).select(
        "product_variant_id",
        F.col("cv").alias(f"cv_{scope_label}"),
    )


cv_store_df = _cv_by_scope(
    monthly_demand_df.filter(F.col("location_type") == F.lit(LOCATION_TYPE_STORE)),
    LOCATION_TYPE_STORE,
)
cv_warehouse_df = _cv_by_scope(
    monthly_demand_df.filter(F.col("location_type") == F.lit(LOCATION_TYPE_WAREHOUSE)),
    LOCATION_TYPE_WAREHOUSE,
)
cv_all_df = overall_avg_and_cv_all_df.select(
    "product_variant_id", F.col("cv").alias("cv_all")
)

location_scope_check_df = cv_all_df.join(
    cv_store_df, "product_variant_id", "left"
).join(cv_warehouse_df, "product_variant_id", "left")

show_result(location_scope_check_df.orderBy("product_variant_id"))

# COMMAND ----------

print("SEASONALITY RANKING COMPLETE")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Final table contract (join with dim_product)
# MAGIC
# MAGIC `seasonality_month_df` does not include `sku` or `product_name`; these come
# MAGIC from `dim_product`, joined by `product_variant_id`, the same key used by the
# MAGIC other joins in this notebook. `product_seasonality_df` is the final contract
# MAGIC for `agg_product_seasonality`, at the `product_variant_id` + `month_number` grain.

# COMMAND ----------

product_attributes_df = spark.table(PRODUCT_TABLE).select(  # noqa: F821
    "product_variant_id", "sku", "product_name"
)

product_attributes_row_count = product_attributes_df.count()
product_attributes_distinct_id_count = (
    product_attributes_df.select("product_variant_id").distinct().count()
)
print(f"Rows in product_attributes_df          : {product_attributes_row_count}")
print(
    "product_variant_id distintos em product_attributes_df : "
    f"{product_attributes_distinct_id_count}"
)
if product_attributes_row_count != product_attributes_distinct_id_count:
    raise AssertionError(
        "dim_product has duplicate product_variant_id values; the join is unsafe: "
        f"{product_attributes_row_count} row(s) versus "
        f"{product_attributes_distinct_id_count} distinct product_variant_id value(s)."
    )
print("dim_product has at most one row per product_variant_id.")

product_seasonality_df = seasonality_month_df.join(
    product_attributes_df, "product_variant_id", "inner"
).select(
    "product_variant_id",
    "sku",
    "product_name",
    "month_number",
    "avg_monthly_demand_qty",
    "overall_avg_monthly_demand_qty",
    "seasonality_index",
    "seasonality_class",
    "observations",
)

seasonality_month_row_count = seasonality_month_df.count()
product_seasonality_row_count = product_seasonality_df.count()
print(f"Rows in seasonality_month_df    : {seasonality_month_row_count}")
print(f"Rows in product_seasonality_df  : {product_seasonality_row_count}")
if product_seasonality_row_count != seasonality_month_row_count:
    raise AssertionError(
        "Joining dim_product changed the row count (expected to match "
        "seasonality_month_df without adding or dropping rows): "
        f"seasonality_month_df={seasonality_month_row_count} vs "
        f"product_seasonality_df={product_seasonality_row_count}"
    )
print("Row count preserved by the dim_product join.")

show_result(
    product_seasonality_df.orderBy("product_variant_id", "month_number"), rows=200
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Persist monthly seasonality
# MAGIC
# MAGIC Follow the pattern in `04_location_forecast_growth.py`: the first run
# MAGIC creates the Delta table in `append` mode; later runs use `MERGE` on the
# MAGIC `product_variant_id + month_number` key to update existing records and
# MAGIC insert new ones.

# COMMAND ----------

if spark.catalog.tableExists(SEASONALITY_TABLE):  # noqa: F821
    from delta.tables import DeltaTable

    target = DeltaTable.forName(spark, SEASONALITY_TABLE)  # noqa: F821
    merge_condition = (
        "target.product_variant_id = source.product_variant_id "
        "AND target.month_number = source.month_number"
    )
    (
        target.alias("target")
        .merge(product_seasonality_df.alias("source"), merge_condition)
        .whenMatchedUpdateAll()
        .whenNotMatchedInsertAll()
        .execute()
    )
else:
    # Append also creates the table on the first run and works in Serverless.
    product_seasonality_df.write.format("delta").mode("append").saveAsTable(
        SEASONALITY_TABLE
    )

persisted_seasonality_df = spark.table(SEASONALITY_TABLE)  # noqa: F821
show_result(
    persisted_seasonality_df.orderBy("product_variant_id", "month_number"), rows=200
)

print(f"Seasonality persisted to: {SEASONALITY_TABLE}")
print("SEASONALITY PERSISTENCE COMPLETE")
