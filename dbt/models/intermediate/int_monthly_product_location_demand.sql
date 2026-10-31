{{
    config(
        materialized='view',
        tags=['forecast', 'demand']
    )
}}

with params as (
    select {{ forecast_cutoff_date() }} as cutoff_date
),

-- =====================================================================
-- STEP 1: Sales scope (official demand rule).
-- =====================================================================
sales_scope as (
    select
        trunc(cast(s.sale_datetime as date), 'month') as demand_month,
        s.product_variant_id,
        l.location_id,
        s.order_id,
        s.sale_datetime,
        s.quantity_qty,
        s.gross_revenue_brl
    from {{ ref('fct_sales') }} s
    inner join {{ ref('dim_location') }} l
        on s.location_key = l.location_key
    cross join params p
    where s.order_status = 'paid'
      and cast(s.sale_datetime as date) <= p.cutoff_date
),

-- =====================================================================
-- STEP 2: Actual demand, aggregated at monthly grain.
-- =====================================================================
realized_demand as (
    select
        demand_month,
        product_variant_id,
        location_id,
        sum(quantity_qty) as demand_qty,
        sum(gross_revenue_brl) as gross_revenue_brl,
        count(distinct order_id) as order_count,
        count(*) as item_line_count,
        count(distinct case when quantity_qty > 0 then cast(sale_datetime as date) end) as days_with_sales
    from sales_scope
    group by
        demand_month,
        product_variant_id,
        location_id
),

-- =====================================================================
-- STEP 3: Series to complete (product-location pair, first month, and
-- static product and location dimensions).
-- =====================================================================
series_base as (
    select
        product_variant_id,
        location_id,
        min(demand_month) as first_demand_month
    from realized_demand
    group by
        product_variant_id,
        location_id
),

series as (
    select
        sb.product_variant_id,
        sb.location_id,
        sb.first_demand_month,
        p.product_key,
        p.sku,
        p.product_name,
        p.category_name,
        p.brand_name,
        l.location_key,
        l.location_name,
        l.location_type,
        l.city,
        l.state
    from series_base sb
    inner join {{ ref('dim_product') }} p
        on sb.product_variant_id = p.product_variant_id
    inner join {{ ref('dim_location') }} l
        on sb.location_id = l.location_id
),

-- =====================================================================
-- STEP 4: Monthly calendar through the cutoff, with year/quarter/month
-- attributes from dim_date (any day in the month works; all three
-- attributes are constant within a month).
-- =====================================================================
calendar_spine as (
    select distinct
        trunc(d.full_date, 'month') as demand_month,
        d.year_number,
        d.quarter_number,
        d.month_number
    from {{ ref('dim_date') }} d
    cross join params p
    where d.full_date <= p.cutoff_date
),

-- =====================================================================
-- STEP 5: Complete grid (series x month, from the first month through the cutoff).
-- =====================================================================
complete_grid as (
    select
        m.demand_month,
        m.year_number,
        m.quarter_number,
        m.month_number,
        s.product_variant_id,
        s.product_key,
        s.sku,
        s.product_name,
        s.category_name,
        s.brand_name,
        s.location_id,
        s.location_key,
        s.location_name,
        s.location_type,
        s.city,
        s.state
    from series s
    inner join calendar_spine m
        on s.first_demand_month <= m.demand_month
),

-- =====================================================================
-- STEP 6: Series eligibility for regression.
-- =====================================================================
series_eligibility as (
    select
        g.product_variant_id,
        g.location_id,
        count(*) as history_month_count,
        sum(
            case
                when coalesce(r.demand_qty, 0) > 0 then 1
                else 0
            end
        ) as positive_demand_month_count
    from complete_grid g
    left join realized_demand r
        on g.demand_month = r.demand_month
        and g.product_variant_id = r.product_variant_id
        and g.location_id = r.location_id
    group by
        g.product_variant_id,
        g.location_id
),

-- =====================================================================
-- STEP 7: Fill metrics with zero for months without sales.
-- =====================================================================
final as (
    select
        g.demand_month,
        g.year_number,
        g.quarter_number,
        g.month_number,
        g.product_variant_id,
        g.product_key,
        g.sku,
        g.product_name,
        g.category_name,
        g.brand_name,
        g.location_id,
        g.location_key,
        g.location_name,
        g.location_type,
        g.city,
        g.state,
        coalesce(r.demand_qty, 0) as demand_qty,
        coalesce(r.gross_revenue_brl, 0) as gross_revenue_brl,
        coalesce(r.order_count, 0) as order_count,
        coalesce(r.item_line_count, 0) as item_line_count,
        coalesce(r.days_with_sales, 0) as days_with_sales,
        coalesce(r.demand_qty, 0) > 0 as has_positive_demand,
        e.history_month_count,
        e.positive_demand_month_count,
        e.history_month_count >= 6
            and e.positive_demand_month_count >= 3 as is_regression_eligible,
        case
            when e.history_month_count >= 6
                and e.positive_demand_month_count >= 3 then 'regression'
            else 'baseline_fallback'
        end as forecast_strategy
    from complete_grid g
    left join realized_demand r
        on g.demand_month = r.demand_month
        and g.product_variant_id = r.product_variant_id
        and g.location_id = r.location_id
    inner join series_eligibility e
        on g.product_variant_id = e.product_variant_id
        and g.location_id = e.location_id
)

select * from final
