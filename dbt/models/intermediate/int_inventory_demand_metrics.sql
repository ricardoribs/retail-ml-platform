with params as (
    select
        {{ var('inventory_config')['demand_window_30d'] }} as window_30d,
        {{ var('inventory_config')['demand_window_60d'] }} as window_60d,
        {{ var('inventory_config')['demand_window_days'] }} as window_90d,
        {{ var('forecast_config')['reference_date_sql'] }} as reference_date
),

daily_demand as (
    select * from {{ ref('int_daily_demand') }}
),

demand_metrics as (
    select
        dd.product_variant_id,
        dd.location_id,
        
        sum(case 
            when dd.sale_date > dateadd(day, -p.window_30d, p.reference_date) 
            then dd.daily_demand_qty 
            else 0 
        end) as demand_30d_qty,
        
        sum(case 
            when dd.sale_date > dateadd(day, -p.window_60d, p.reference_date) 
            then dd.daily_demand_qty 
            else 0 
        end) as demand_60d_qty,
        
        sum(dd.daily_demand_qty) as demand_90d_qty,
        
        sum(case 
            when dd.sale_date > dateadd(day, -p.window_30d, p.reference_date) 
            then dd.daily_demand_qty 
            else 0 
        end) / p.window_30d as avg_daily_demand_30d,
        
        sum(case 
            when dd.sale_date > dateadd(day, -p.window_60d, p.reference_date) 
            then dd.daily_demand_qty 
            else 0 
        end) / p.window_60d as avg_daily_demand_60d,
        
        sum(dd.daily_demand_qty) / p.window_90d as avg_daily_demand_90d,
        
        stddev(dd.daily_demand_qty) as daily_demand_stddev_90d,
        
        sum(case when dd.daily_demand_qty > 0 then 1 else 0 end) as days_with_sales_90d,
        
        case 
            when sum(dd.daily_demand_qty) > 0 then true 
            else false 
        end as has_recent_sales
        
    from daily_demand dd
    cross join params p
    group by 
        dd.product_variant_id,
        dd.location_id,
        p.window_30d,
        p.window_60d,
        p.window_90d,
        p.reference_date
)

select * from demand_metrics
