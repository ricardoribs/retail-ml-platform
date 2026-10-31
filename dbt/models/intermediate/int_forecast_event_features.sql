{{ config(materialized='view', tags=['forecast', 'features', 'research']) }}

-- Inputs always precede demand_month. Current demand is only the target.
with monthly as (
    select
        demand_month,
        product_variant_id,
        location_id,
        location_type,
        category_name,
        month_number,
        demand_qty,
        days_with_sales,
        order_count,
        months_available_before_target
    from {{ ref('int_forecast_feature_matrix') }}
),
monthly_price as (
    select
        trunc(cast(s.sale_datetime as date), 'month') as sale_month,
        s.product_variant_id,
        l.location_id,
        sum(s.unit_price * s.quantity_qty) / nullif(sum(s.quantity_qty), 0) as unit_price_brl
    from {{ ref('fct_sales') }} s
    inner join {{ ref('dim_location') }} l on s.location_key = l.location_key
    where s.order_status = 'paid' and s.quantity_qty > 0
    group by 1, 2, 3
),
joined as (
    select
        m.*,
        p.unit_price_brl
    from monthly m
    left join monthly_price p
        on m.demand_month = p.sale_month
        and m.product_variant_id = p.product_variant_id
        and m.location_id = p.location_id
),
lagged as (
    select
        *,
        last(unit_price_brl, true) over (
            partition by product_variant_id, location_id
            order by demand_month rows between unbounded preceding and 1 preceding
        ) as last_observed_unit_price_brl,
        lag(unit_price_brl) over (
            partition by product_variant_id, location_id order by demand_month
        ) as previous_month_unit_price_brl,
        lag(demand_qty) over (
            partition by product_variant_id, location_id order by demand_month
        ) as previous_month_demand_qty,
        sum(case when demand_qty > 0 then 1 else 0 end) over (
            partition by product_variant_id, location_id
            order by demand_month rows between 3 preceding and 1 preceding
        ) as positive_months_last_3,
        sum(case when demand_qty > 0 then 1 else 0 end) over (
            partition by product_variant_id, location_id
            order by demand_month rows between 12 preceding and 1 preceding
        ) as positive_months_last_12,
        sum(days_with_sales) over (
            partition by product_variant_id, location_id
            order by demand_month rows between 3 preceding and 1 preceding
        ) as sales_days_last_3,
        sum(order_count) over (
            partition by product_variant_id, location_id
            order by demand_month rows between 3 preceding and 1 preceding
        ) as orders_last_3
    from joined
)
select
    demand_month,
    product_variant_id,
    location_id,
    location_type,
    category_name,
    month_number,
    demand_qty,
    months_available_before_target,
    last_observed_unit_price_brl,
    previous_month_unit_price_brl,
    previous_month_demand_qty,
    positive_months_last_3,
    positive_months_last_12,
    sales_days_last_3,
    orders_last_3,
    case when last_observed_unit_price_brl is null then 0 else 1 end as has_price_history
from lagged
