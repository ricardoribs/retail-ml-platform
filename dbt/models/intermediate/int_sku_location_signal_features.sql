{{
    config(
        materialized='view',
        tags=['forecast', 'features', 'sku_location']
    )
}}

-- Signals known before the forecast month. Current catalog/balance fields are
-- deliberately excluded because they have no reliable as-of history.
with monthly_demand as (
    select *
    from {{ ref('int_monthly_product_location_demand') }}
),

received_by_month as (
    select
        trunc(cast(p.received_at as date), 'month') as receipt_month,
        product.product_variant_id,
        location.location_id,
        sum(p.quantity_received_qty) as received_qty
    from {{ ref('fct_purchases') }} p
    inner join {{ ref('dim_product') }} product
        on p.product_key = product.product_key
    inner join {{ ref('dim_location') }} location
        on p.location_key = location.location_key
    group by
        trunc(cast(p.received_at as date), 'month'),
        product.product_variant_id,
        location.location_id
),

monthly_signals as (
    select
        demand.*,
        case
            when demand.demand_qty > 0
                then demand.gross_revenue_brl / demand.demand_qty
        end as realized_unit_price,
        coalesce(received.received_qty, 0) as received_qty
    from monthly_demand demand
    left join received_by_month received
        on demand.demand_month = received.receipt_month
        and demand.product_variant_id = received.product_variant_id
        and demand.location_id = received.location_id
),

as_of_features as (
    select
        *,
        last_value(realized_unit_price, true) over (
            partition by product_variant_id, location_id
            order by demand_month
            rows between unbounded preceding and 1 preceding
        ) as last_unit_price,
        coalesce(
            sum(days_with_sales) over (
                partition by product_variant_id, location_id
                order by demand_month
                rows between 3 preceding and 1 preceding
            ),
            0
        ) as rolling_3m_days_with_sales,
        coalesce(
            sum(case when demand_qty > 0 then 1 else 0 end) over (
                partition by product_variant_id, location_id
                order by demand_month
                rows between 6 preceding and 1 preceding
            ),
            0
        ) as rolling_6m_positive_months,
        max(case when demand_qty > 0 then demand_month end) over (
            partition by product_variant_id, location_id
            order by demand_month
            rows between unbounded preceding and 1 preceding
        ) as last_positive_demand_month,
        coalesce(
            sum(received_qty) over (
                partition by product_variant_id, location_id
                order by demand_month
                rows between 3 preceding and 1 preceding
            ),
            0
        ) as received_qty_last_3m
    from monthly_signals
)

select
    demand_month,
    year_number,
    quarter_number,
    month_number,
    product_variant_id,
    product_key,
    sku,
    product_name,
    category_name,
    brand_name,
    location_id,
    location_key,
    location_name,
    location_type,
    city,
    state,
    demand_qty,
    gross_revenue_brl,
    order_count,
    item_line_count,
    days_with_sales,
    has_positive_demand,
    history_month_count,
    positive_demand_month_count,
    is_regression_eligible,
    forecast_strategy,
    realized_unit_price,
    received_qty,
    last_unit_price,
    rolling_3m_days_with_sales,
    rolling_6m_positive_months,
    last_positive_demand_month,
    received_qty_last_3m,
    case
        when last_positive_demand_month is not null
            then floor(months_between(demand_month, last_positive_demand_month))
    end as months_since_last_positive_sale
from as_of_features
