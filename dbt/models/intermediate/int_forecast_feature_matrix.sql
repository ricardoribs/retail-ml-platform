{{
    config(
        materialized='view',
        tags=['forecast', 'features']
    )
}}

with monthly_demand as (
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
        last_unit_price,
        rolling_3m_days_with_sales,
        rolling_6m_positive_months,
        months_since_last_positive_sale,
        received_qty_last_3m
    from {{ ref('int_sku_location_signal_features') }}
),

window_features as (
    select
        *,
        row_number() over (
            partition by product_variant_id, location_id
            order by demand_month
        ) - 1 as months_available_before_target,
        lag(demand_qty, 1) over (
            partition by product_variant_id, location_id
            order by demand_month
        ) as lag_1_qty,
        lag(demand_qty, 2) over (
            partition by product_variant_id, location_id
            order by demand_month
        ) as lag_2_qty,
        lag(demand_qty, 3) over (
            partition by product_variant_id, location_id
            order by demand_month
        ) as lag_3_qty,
        avg(demand_qty) over (
            partition by product_variant_id, location_id
            order by demand_month
            rows between 3 preceding and 1 preceding
        ) as rolling_3m_avg_raw,
        avg(demand_qty) over (
            partition by product_variant_id, location_id
            order by demand_month
            rows between 6 preceding and 1 preceding
        ) as rolling_6m_avg_raw,
        sum(demand_qty) over (
            partition by product_variant_id, location_id
            order by demand_month
            rows between 3 preceding and 1 preceding
        ) as rolling_3m_sum_raw
    from monthly_demand
),

final as (
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
        last_unit_price,
        rolling_3m_days_with_sales,
        rolling_6m_positive_months,
        months_since_last_positive_sale,
        received_qty_last_3m,
        lag_1_qty,
        lag_2_qty,
        lag_3_qty,
        case
            when months_available_before_target >= 3 then rolling_3m_avg_raw
        end as rolling_3m_avg_qty,
        case
            when months_available_before_target >= 6 then rolling_6m_avg_raw
        end as rolling_6m_avg_qty,
        case
            when months_available_before_target >= 3 then rolling_3m_sum_raw
        end as rolling_3m_sum_qty,
        gross_revenue_brl,
        order_count,
        item_line_count,
        days_with_sales,
        has_positive_demand,
        history_month_count,
        positive_demand_month_count,
        months_available_before_target,
        is_regression_eligible,
        is_regression_eligible
            and months_available_before_target >= 6 as is_training_eligible,
        forecast_strategy
    from window_features
)

select * from final
