# Databricks notebook source

# COMMAND ----------

"""Initial data audit for forecasting.

This notebook answers the questions that must be resolved before predictive modeling:
1. What is the actual time coverage of orders and order items?
2. Which order statuses exist, and which appear to represent valid demand?
3. Is the product-store grain (`product_variant_id + location_id`) available?
4. How might returns and cancellations affect the demand rule?
5. Are there enough locations and distribution centers for the analyses?"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

from pyspark.sql import DataFrame

if TYPE_CHECKING:
    dbutils: Any
    display: Any
    spark: Any

# COMMAND ----------

try:
    dbutils.widgets.text("catalog", "retail_analytics")  # noqa: F821
    dbutils.widgets.text("staging_schema", "default_staging")  # noqa: F821
    dbutils.widgets.text("intermediate_schema", "default_intermediate")  # noqa: F821
    dbutils.widgets.text("marts_schema", "default_marts")  # noqa: F821
    dbutils.widgets.text("reference_date", "current_date()")  # noqa: F821

    CATALOG = dbutils.widgets.get("catalog")  # noqa: F821
    STAGING_SCHEMA = dbutils.widgets.get("staging_schema")  # noqa: F821
    INTERMEDIATE_SCHEMA = dbutils.widgets.get("intermediate_schema")  # noqa: F821
    MARTS_SCHEMA = dbutils.widgets.get("marts_schema")  # noqa: F821
    REFERENCE_DATE = dbutils.widgets.get("reference_date")  # noqa: F821
except NameError:
    CATALOG = "retail_analytics"
    STAGING_SCHEMA = "default_staging"
    INTERMEDIATE_SCHEMA = "default_intermediate"
    MARTS_SCHEMA = "default_marts"
    REFERENCE_DATE = "current_date()"


def fq_table(schema: str, table: str) -> str:
    """Return the fully qualified name for Unity Catalog."""
    return f"{CATALOG}.{schema}.{table}"


STG_ORDERS = fq_table(STAGING_SCHEMA, "stg_orders")
STG_ORDER_ITEMS = fq_table(STAGING_SCHEMA, "stg_order_items")
STG_RETURNS = fq_table(STAGING_SCHEMA, "stg_returns")
STG_RETURN_ITEMS = fq_table(STAGING_SCHEMA, "stg_return_items")
INT_DAILY_DEMAND = fq_table(INTERMEDIATE_SCHEMA, "int_daily_demand")
FCT_ORDERS = fq_table(MARTS_SCHEMA, "fct_orders")
FCT_SALES = fq_table(MARTS_SCHEMA, "fct_sales")
FCT_RETURNS = fq_table(MARTS_SCHEMA, "fct_returns")
FCT_STOCK_MOVEMENTS = fq_table(MARTS_SCHEMA, "fct_stock_movements")
DIM_PRODUCT = fq_table(MARTS_SCHEMA, "dim_product")
DIM_LOCATION = fq_table(MARTS_SCHEMA, "dim_location")

print("Audit parameters")
print(f"  Catalog             : {CATALOG}")
print(f"  Schema staging      : {STAGING_SCHEMA}")
print(f"  Schema intermediate : {INTERMEDIATE_SCHEMA}")
print(f"  Schema marts        : {MARTS_SCHEMA}")
print(f"  Reference date      : {REFERENCE_DATE}")

# COMMAND ----------


def show_result(df: DataFrame, rows: int = 100) -> None:
    """Display a DataFrame in Databricks, with a local execution fallback."""
    try:
        display(df)  # noqa: F821
    except NameError:
        df.show(rows, truncate=False)


def table_exists(schema: str, table_name: str) -> bool:
    result = spark.sql(  # noqa: F821
        f"SHOW TABLES IN {CATALOG}.{schema} LIKE '{table_name}'"
    )
    return result.count() > 0


@dataclass(frozen=True)
class TableCheck:
    schema: str
    table_name: str
    purpose: str

    @property
    def full_name(self) -> str:
        return fq_table(self.schema, self.table_name)


def run_query(title: str, sql_text: str, rows: int = 100) -> DataFrame:
    print(f"\n=== {title} ===")
    print(sql_text)
    try:
        df = spark.sql(sql_text)  # noqa: F821
        show_result(df, rows=rows)
        return df
    except Exception as exc:
        error_message = str(exc)
        print(f"[ERROR] Section failed: {title}")
        print(error_message)
        return spark.createDataFrame(  # noqa: F821
            [{"section": title, "error": error_message}]
        )


def audit_table_inventory(checks: Iterable[TableCheck]) -> DataFrame:
    rows = []
    for check in checks:
        exists = table_exists(check.schema, check.table_name)
        row_count = None
        if exists:
            row_count = spark.sql(  # noqa: F821
                f"select count(*) as row_count from {check.full_name}"
            ).collect()[0]["row_count"]

        rows.append(
            {
                "schema_name": check.schema,
                "table_name": check.table_name,
                "full_name": check.full_name,
                "exists": exists,
                "row_count": row_count,
                "purpose": check.purpose,
            }
        )

    return spark.createDataFrame(rows)  # noqa: F821


TABLE_CHECKS = [
    TableCheck(STAGING_SCHEMA, "stg_orders", "status e datas de pedidos"),
    TableCheck(STAGING_SCHEMA, "stg_order_items", "itens e quantidades vendidas"),
    TableCheck(STAGING_SCHEMA, "stg_returns", "return statuses and dates"),
    TableCheck(STAGING_SCHEMA, "stg_return_items", "quantidades devolvidas"),
    TableCheck(
        INTERMEDIATE_SCHEMA,
        "int_daily_demand",
        "legacy daily demand rule",
    ),
    TableCheck(MARTS_SCHEMA, "fct_orders", "enriched orders and eligibility"),
    TableCheck(MARTS_SCHEMA, "fct_sales", "fato de vendas por item"),
    TableCheck(MARTS_SCHEMA, "fct_returns", "return facts by item"),
    TableCheck(MARTS_SCHEMA, "fct_stock_movements", "inventory movements"),
    TableCheck(MARTS_SCHEMA, "dim_product", "product/SKU records"),
    TableCheck(MARTS_SCHEMA, "dim_location", "store and distribution center records"),
]

# COMMAND ----------

inventory_df = audit_table_inventory(TABLE_CHECKS)
show_result(inventory_df.orderBy("schema_name", "table_name"))

missing_tables = [
    row["full_name"]
    for row in inventory_df.filter("exists = false").select("full_name").collect()
]

if missing_tables:
    print("[WARNING] Tables missing from the audit:")
    for table in missing_tables:
        print(f"  - {table}")
else:
    print("[OK] All expected tables exist.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Time coverage and volume
# MAGIC
# MAGIC Confirm that enough history is available for:
# MAGIC
# MAGIC - a three-month forecast;
# MAGIC - seasonality calculations;
# MAGIC - growth by store/POS;
# MAGIC - distribution center growth.

# COMMAND ----------

run_query(
    "Cobertura temporal em fct_sales",
    f"""
    select
        min(cast(sale_datetime as date)) as first_sale_date,
        max(cast(sale_datetime as date)) as last_sale_date,
        count(*) as sales_item_rows,
        count(distinct order_id) as order_count,
        count(distinct product_variant_id) as product_variant_count,
        count(distinct location_key) as location_key_count,
        sum(quantity_qty) as total_quantity_qty,
        sum(line_total) as total_line_amount_brl
    from {FCT_SALES}
    """,
)

# COMMAND ----------

run_query(
    "Monthly sales volume",
    f"""
    select
        date_trunc('month', cast(sale_datetime as date)) as sale_month,
        count(*) as sales_item_rows,
        count(distinct order_id) as order_count,
        count(distinct product_variant_id) as product_variant_count,
        count(distinct location_key) as location_key_count,
        sum(quantity_qty) as total_quantity_qty,
        sum(line_total) as total_line_amount_brl
    from {FCT_SALES}
    group by date_trunc('month', cast(sale_datetime as date))
    order by sale_month
    """,
    rows=200,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Order status and candidate demand rules
# MAGIC
# MAGIC This section identifies the available statuses and compares three
# MAGIC candidate rules:
# MAGIC
# MAGIC - `paid_only`: only `order_status = 'paid'`;
# MAGIC - `completed_family`: commercially completed statuses;
# MAGIC - `kpi_eligible`: orders marked as eligible in `fct_orders`.
# MAGIC
# MAGIC Finalize the demand rule before modeling.

# COMMAND ----------

run_query(
    "Status distribution in fct_sales",
    f"""
    select
        order_status,
        count(*) as sales_item_rows,
        count(distinct order_id) as order_count,
        count(distinct product_variant_id) as product_variant_count,
        count(distinct location_key) as location_key_count,
        min(cast(sale_datetime as date)) as first_sale_date,
        max(cast(sale_datetime as date)) as last_sale_date,
        sum(quantity_qty) as total_quantity_qty,
        sum(line_total) as total_line_amount_brl
    from {FCT_SALES}
    group by order_status
    order by total_quantity_qty desc
    """,
)

# COMMAND ----------

run_query(
    "Eligibility and reconciliation in fct_orders",
    f"""
    select
        order_status,
        payment_reconciliation_status,
        is_eligible_for_kpis_flag,
        count(*) as order_count,
        sum(total_units) as total_units,
        sum(order_total_amount_brl) as order_total_amount_brl,
        sum(recognized_net_revenue_amount_brl) as recognized_net_revenue_amount_brl
    from {FCT_ORDERS}
    group by
        order_status,
        payment_reconciliation_status,
        is_eligible_for_kpis_flag
    order by order_count desc
    """,
)

# COMMAND ----------

run_query(
    "Comparison of candidate demand rules",
    f"""
    with sales_orders as (
        select
            s.order_id,
            s.product_variant_id,
            s.location_key,
            s.quantity_qty,
            s.line_total,
            s.order_status,
            o.is_eligible_for_kpis_flag
        from {FCT_SALES} s
        left join {FCT_ORDERS} o
            on s.order_id = o.order_id
    ),

    candidates as (
        select
            'paid_only' as demand_rule,
            *
        from sales_orders
        where order_status = 'paid'

        union all

        select
            'completed_family' as demand_rule,
            *
        from sales_orders
        where order_status in ('paid', 'completed', 'shipped', 'delivered')

        union all

        select
            'kpi_eligible' as demand_rule,
            *
        from sales_orders
        where is_eligible_for_kpis_flag = true
    )

    select
        demand_rule,
        count(*) as sales_item_rows,
        count(distinct order_id) as order_count,
        count(distinct product_variant_id) as product_variant_count,
        count(distinct location_key) as location_key_count,
        sum(quantity_qty) as total_quantity_qty,
        sum(line_total) as total_line_amount_brl
    from candidates
    group by demand_rule
    order by demand_rule
    """,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Product, location, and forecast-grain coverage
# MAGIC
# MAGIC Plan the forecast at this grain:
# MAGIC
# MAGIC `product_variant_id + location_id + forecast_month`
# MAGIC
# MAGIC This section checks whether the data supports this grain.

# COMMAND ----------

run_query(
    "Distribution by location type",
    f"""
    select
        l.location_type,
        count(distinct l.location_id) as location_count,
        count(*) as sales_item_rows,
        count(distinct s.order_id) as order_count,
        count(distinct s.product_variant_id) as product_variant_count,
        sum(s.quantity_qty) as total_quantity_qty,
        sum(s.line_total) as total_line_amount_brl
    from {FCT_SALES} s
    left join {DIM_LOCATION} l
        on s.location_key = l.location_key
    group by l.location_type
    order by total_quantity_qty desc
    """,
)

# COMMAND ----------

run_query(
    "Coverage by product and location",
    f"""
    with sales_by_pair_month as (
        select
            s.product_variant_id,
            l.location_id,
            l.location_type,
            date_trunc('month', cast(s.sale_datetime as date)) as sale_month,
            sum(s.quantity_qty) as monthly_quantity_qty
        from {FCT_SALES} s
        left join {DIM_LOCATION} l
            on s.location_key = l.location_key
        group by
            s.product_variant_id,
            l.location_id,
            l.location_type,
            date_trunc('month', cast(s.sale_datetime as date))
    ),

    pair_summary as (
        select
            product_variant_id,
            location_id,
            location_type,
            count(*) as months_with_sales,
            sum(monthly_quantity_qty) as total_quantity_qty,
            avg(monthly_quantity_qty) as avg_monthly_quantity_qty
        from sales_by_pair_month
        group by product_variant_id, location_id, location_type
    )

    select
        location_type,
        count(*) as product_location_pair_count,
        percentile_approx(months_with_sales, 0.5) as median_months_with_sales,
        avg(months_with_sales) as avg_months_with_sales,
        sum(case when months_with_sales >= 3 then 1 else 0 end) as pairs_with_3m_plus,
        sum(case when months_with_sales >= 6 then 1 else 0 end) as pairs_with_6m_plus,
        sum(case when months_with_sales >= 12 then 1 else 0 end) as pairs_with_12m_plus,
        sum(total_quantity_qty) as total_quantity_qty
    from pair_summary
    group by location_type
    order by product_location_pair_count desc
    """,
)

# COMMAND ----------

run_query(
    "Top products by volume and monthly history",
    f"""
    with monthly_product as (
        select
            product_variant_id,
            date_trunc('month', cast(sale_datetime as date)) as sale_month,
            sum(quantity_qty) as monthly_quantity_qty
        from {FCT_SALES}
        group by product_variant_id, date_trunc('month', cast(sale_datetime as date))
    ),

    product_summary as (
        select
            product_variant_id,
            count(*) as months_with_sales,
            sum(monthly_quantity_qty) as total_quantity_qty,
            avg(monthly_quantity_qty) as avg_monthly_quantity_qty
        from monthly_product
        group by product_variant_id
    )

    select
        ps.product_variant_id,
        p.sku,
        p.product_name,
        p.category_name,
        p.brand_name,
        ps.months_with_sales,
        ps.total_quantity_qty,
        ps.avg_monthly_quantity_qty
    from product_summary ps
    left join {DIM_PRODUCT} p
        on ps.product_variant_id = p.product_variant_id
    order by ps.total_quantity_qty desc, ps.months_with_sales desc
    limit 50
    """,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Sparse series and model feasibility
# MAGIC
# MAGIC This section measures how many product-location series have enough history
# MAGIC for modeling and how many will need a fallback, aggregation, or baseline.

# COMMAND ----------

run_query(
    "Product-location series with a complete monthly calendar",
    f"""
    with bounds as (
        select
            date_trunc('month', min(cast(sale_datetime as date))) as min_month,
            date_trunc('month', max(cast(sale_datetime as date))) as max_month
        from {FCT_SALES}
    ),

    months as (
        select explode(sequence(min_month, max_month, interval 1 month)) as sale_month
        from bounds
    ),

    pairs as (
        select distinct
            s.product_variant_id,
            l.location_id,
            l.location_type
        from {FCT_SALES} s
        left join {DIM_LOCATION} l
            on s.location_key = l.location_key
    ),

    monthly_sales as (
        select
            s.product_variant_id,
            l.location_id,
            date_trunc('month', cast(s.sale_datetime as date)) as sale_month,
            sum(s.quantity_qty) as monthly_quantity_qty
        from {FCT_SALES} s
        left join {DIM_LOCATION} l
            on s.location_key = l.location_key
        group by
            s.product_variant_id,
            l.location_id,
            date_trunc('month', cast(s.sale_datetime as date))
    ),

    complete_series as (
        select
            p.product_variant_id,
            p.location_id,
            p.location_type,
            m.sale_month,
            coalesce(ms.monthly_quantity_qty, 0) as monthly_quantity_qty
        from pairs p
        cross join months m
        left join monthly_sales ms
            on p.product_variant_id = ms.product_variant_id
            and p.location_id = ms.location_id
            and m.sale_month = ms.sale_month
    ),

    series_summary as (
        select
            product_variant_id,
            location_id,
            location_type,
            count(*) as total_months,
            sum(
                case when monthly_quantity_qty > 0 then 1 else 0 end
            ) as months_with_sales,
            sum(case when monthly_quantity_qty = 0 then 1 else 0 end) as zero_months,
            sum(monthly_quantity_qty) as total_quantity_qty
        from complete_series
        group by product_variant_id, location_id, location_type
    )

    select
        location_type,
        count(*) as series_count,
        avg(months_with_sales) as avg_months_with_sales,
        percentile_approx(months_with_sales, 0.5) as median_months_with_sales,
        sum(
            case when months_with_sales >= 12 then 1 else 0 end
        ) as series_with_12m_plus,
        sum(
            case when months_with_sales between 6 and 11 then 1 else 0 end
        ) as series_with_6_to_11m,
        sum(
            case when months_with_sales between 3 and 5 then 1 else 0 end
        ) as series_with_3_to_5m,
        sum(
            case when months_with_sales < 3 then 1 else 0 end
        ) as sparse_series_under_3m,
        sum(total_quantity_qty) as total_quantity_qty
    from series_summary
    group by location_type
    order by series_count desc
    """,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Returns, cancellations, and net demand
# MAGIC
# MAGIC The demand rule must specify whether the forecast uses gross or net demand.
# MAGIC The initial recommendation is to start with gross demand validated by order
# MAGIC status and document returns as an adjustment or limitation, unless the audit
# MAGIC finds significant returns that can be reliably linked to sold items.

# COMMAND ----------

run_query(
    "Returns summary by status",
    f"""
    select
        count(*) as return_item_rows,
        count(distinct order_id) as returned_order_count,
        count(distinct product_variant_id) as returned_product_variant_count,
        sum(quantity_returned_qty) as total_returned_quantity_qty,
        sum(return_amount) as total_return_amount_brl
    from {FCT_RETURNS}
    """,
)

# COMMAND ----------

run_query(
    "Gross sales versus returns by product",
    f"""
    with sales as (
        select
            product_variant_id,
            sum(quantity_qty) as sold_quantity_qty,
            sum(line_total) as sold_line_amount_brl
        from {FCT_SALES}
        group by product_variant_id
    ),

    returns as (
        select
            product_variant_id,
            sum(quantity_returned_qty) as returned_quantity_qty,
            sum(return_amount) as returned_amount_brl
        from {FCT_RETURNS}
        group by product_variant_id
    )

    select
        s.product_variant_id,
        p.sku,
        p.product_name,
        s.sold_quantity_qty,
        coalesce(r.returned_quantity_qty, 0) as returned_quantity_qty,
        case
            when s.sold_quantity_qty = 0 then null
            else coalesce(r.returned_quantity_qty, 0) / s.sold_quantity_qty
        end as returned_quantity_rate,
        s.sold_line_amount_brl,
        coalesce(r.returned_amount_brl, 0) as returned_amount_brl
    from sales s
    left join returns r
        on s.product_variant_id = r.product_variant_id
    left join {DIM_PRODUCT} p
        on s.product_variant_id = p.product_variant_id
    order by returned_quantity_rate desc nulls last, s.sold_quantity_qty desc
    limit 50
    """,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Key quality, nulls, and invalid values
# MAGIC
# MAGIC This section looks for issues that could break dbt, feature generation, or training.

# COMMAND ----------

run_query(
    "Basic issues in fct_sales",
    f"""
    select
        count(*) as sales_item_rows,
        sum(
            case when order_item_id is null then 1 else 0 end
        ) as null_order_item_id_rows,
        sum(case when order_id is null then 1 else 0 end) as null_order_id_rows,
        sum(
            case when product_variant_id is null then 1 else 0 end
        ) as null_product_variant_id_rows,
        sum(case when product_key is null then 1 else 0 end) as null_product_key_rows,
        sum(case when location_key is null then 1 else 0 end) as null_location_key_rows,
        sum(
            case when sale_datetime is null then 1 else 0 end
        ) as null_sale_datetime_rows,
        sum(case when quantity_qty is null then 1 else 0 end) as null_quantity_rows,
        sum(case when quantity_qty < 0 then 1 else 0 end) as negative_quantity_rows,
        sum(case when unit_price < 0 then 1 else 0 end) as negative_unit_price_rows,
        sum(case when line_total < 0 then 1 else 0 end) as negative_line_total_rows
    from {FCT_SALES}
    """,
)

# COMMAND ----------

run_query(
    "Duplicidades de chaves principais",
    f"""
    with duplicate_order_items as (
        select 'fct_sales.order_item_id' as check_name, count(*) as duplicate_count
        from (
            select order_item_id
            from {FCT_SALES}
            group by order_item_id
            having count(*) > 1
        )
    ),

    duplicate_products as (
        select
            'dim_product.product_variant_id' as check_name,
            count(*) as duplicate_count
        from (
            select product_variant_id
            from {DIM_PRODUCT}
            group by product_variant_id
            having count(*) > 1
        )
    ),

    duplicate_locations as (
        select 'dim_location.location_id' as check_name, count(*) as duplicate_count
        from (
            select location_id
            from {DIM_LOCATION}
            group by location_id
            having count(*) > 1
        )
    )

    select * from duplicate_order_items
    union all
    select * from duplicate_products
    union all
    select * from duplicate_locations
    """,
)

# COMMAND ----------

run_query(
    "Records without a matching dimension",
    f"""
    select
        sum(
            case when p.product_variant_id is null then 1 else 0 end
        ) as sales_without_product_dim,
        sum(
            case when l.location_key is null then 1 else 0 end
        ) as sales_without_location_dim,
        count(*) as sales_item_rows
    from {FCT_SALES} s
    left join {DIM_PRODUCT} p
        on s.product_variant_id = p.product_variant_id
    left join {DIM_LOCATION} l
        on s.location_key = l.location_key
    """,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Distribution centers and candidate locations
# MAGIC
# MAGIC This section checks whether `location_type` supports direct distribution
# MAGIC center analysis. If distribution centers have no direct demand, the analysis
# MAGIC should use a documented geographic proxy.

# COMMAND ----------

run_query(
    "Locations by type, state, and city",
    f"""
    select
        location_type,
        state,
        city,
        count(*) as location_count,
        sum(case when is_active_flag = true then 1 else 0 end) as active_location_count
    from {DIM_LOCATION}
    group by location_type, state, city
    order by location_type, state, city
    """,
    rows=300,
)

# COMMAND ----------

run_query(
    "Inventory movements by type",
    f"""
    select
        m.movement_type,
        count(*) as movement_rows,
        count(distinct p.product_variant_id) as product_variant_count,
        count(distinct l.location_id) as location_count,
        min(cast(m.occurred_at as date)) as first_movement_date,
        max(cast(m.occurred_at as date)) as last_movement_date,
        sum(m.quantity_moved_qty) as total_quantity_moved_qty
    from {FCT_STOCK_MOVEMENTS} m
    left join {DIM_PRODUCT} p
        on m.product_key = p.product_key
    left join {DIM_LOCATION} l
        on m.location_key = l.location_key
    group by m.movement_type
    order by movement_rows desc
    """,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Summary of modeling decisions
# MAGIC
# MAGIC Fill in these items after running the queries:
# MAGIC
# MAGIC - Official order statuses for demand:
# MAGIC - Demand target field:
# MAGIC - Return rule:
# MAGIC - Official forecast grain:
# MAGIC - Product-location eligibility for training:
# MAGIC - Handling of sparse series:
# MAGIC - Final schema/table for outputs:
# MAGIC - Gaps or required changes before the feature matrix:

# COMMAND ----------

print("DATA AUDIT COMPLETE")
