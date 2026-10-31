{{
    config(
        materialized='view',
        tags=['inventory', 'demand']
    )
}}

-- =====================================================================
-- int_daily_demand
-- =====================================================================
-- DESCRIPTION:
--   Historical daily demand by product and location.
--   Includes EVERY DAY in the window (even with zero sales) for correct
--   calculation of averages and standard deviation.
--
-- GRAIN: sale_date + product_variant_id + location_id
--
-- BUSINESS RULES:
--   - Includes only orders with status 'paid' (gross demand in the MVP).
--   - Configurable window via var('inventory_config')['demand_window_days'].
--   - Reference date via var('forecast_config')['reference_date_sql'].
--   - Days without sales = 0 (not omitted).
--   - Includes only actual data through the reference date.
--
-- DEPENDENCIES:
--   - stg_orders
--   - stg_order_items
--
-- USED BY:
--   - int_inventory_demand_metrics
-- =====================================================================

with params as (
    select
        {{ var('forecast_config')['reference_date_sql'] }} as reference_date,
        {{ var('inventory_config')['demand_window_days'] }} as window_days
),

-- =====================================================================
-- STEP 1: Use the existing dim_date to create the complete calendar.
-- =====================================================================
date_spine as (
    select full_date as sale_date
    from {{ ref('dim_date') }}
),

filtered_dates as (
    select date_spine.sale_date
    from date_spine
    cross join params p
    where date_spine.sale_date > dateadd(day, -p.window_days, p.reference_date)
      and date_spine.sale_date <= p.reference_date
),

-- =====================================================================
-- STEP 2: Identify all product + location pairs.
-- Includes products with sales and products in stock without sales.
-- =====================================================================
product_location_from_sales as (
    select distinct
        oi.product_variant_id,
        o.location_id
    from {{ ref('stg_order_items') }} oi
    join {{ ref('stg_orders') }} o 
        on oi.order_id = o.order_id
    cross join params p
    where o.order_status = 'paid'
      and cast(o.placed_at as date) > dateadd(day, -p.window_days, p.reference_date)
      and cast(o.placed_at as date) <= p.reference_date
),

product_location_from_stock as (
    select distinct
        product_variant_id,
        location_id
    from {{ ref('stg_stock_levels') }}
),

product_location_pairs as (
    select * from product_location_from_sales
    union
    select * from product_location_from_stock
),

-- =====================================================================
-- STEP 3: Calculate actual sales by day.
-- =====================================================================
realized_sales as (
    select
        cast(o.placed_at as date) as sale_date,
        oi.product_variant_id,
        o.location_id,
        sum(oi.quantity_qty) as daily_demand_qty
    from {{ ref('stg_orders') }} o
    join {{ ref('stg_order_items') }} oi 
        on o.order_id = oi.order_id
    cross join params p
    where o.order_status = 'paid'
      and cast(o.placed_at as date) <= p.reference_date
      and cast(o.placed_at as date) > dateadd(day, -p.window_days, p.reference_date)
    group by 
        cast(o.placed_at as date),
        oi.product_variant_id,
        o.location_id
),

-- =====================================================================
-- STEP 4: Create the complete calendar (product x location x date).
-- =====================================================================
complete_calendar as (
    select
        fd.sale_date,
        plp.product_variant_id,
        plp.location_id
    from filtered_dates fd
    cross join product_location_pairs plp
),

-- =====================================================================
-- STEP 5: Fill days without sales with zero.
-- =====================================================================
final as (
    select
        cc.sale_date,
        cc.product_variant_id,
        cc.location_id,
        coalesce(rs.daily_demand_qty, 0) as daily_demand_qty,
        
        -- Flag indicating whether this is realized sales or a projection.
        case 
            when cc.sale_date <= (select p.reference_date from params p)
            then true 
            else false 
        end as is_realized,
        
        -- Flag indicating whether there were sales on this day.
        case 
            when rs.daily_demand_qty > 0 then true 
            else false 
        end as has_sales_on_date
        
    from complete_calendar cc
    left join realized_sales rs
        on cc.sale_date = rs.sale_date
        and cc.product_variant_id = rs.product_variant_id
        and cc.location_id = rs.location_id
)

select * from final
