with params as (
    select 
        {{ var('inventory_config')['service_level_z_score'] }} as z_score
),

stock as (
    select
        product_variant_id,
        location_id,
        quantity_on_hand as current_stock_qty,
        reorder_point as source_reorder_point
    from {{ ref('stg_stock_levels') }}
),

demand as (
    select * from {{ ref('int_inventory_demand_metrics') }}
),

suppliers as (
    select * from {{ ref('int_preferred_suppliers') }}
),

calculated as (
    select
        s.product_variant_id,
        s.location_id,
        s.current_stock_qty,
        s.source_reorder_point,
        
        d.avg_daily_demand_90d,
        d.daily_demand_stddev_90d,
        d.has_recent_sales,
        
        coalesce(sup.lead_time_days, 0) as lead_time_days,
        
        d.avg_daily_demand_90d * coalesce(sup.lead_time_days, 0) as lead_time_demand,
        
        coalesce(
            case
                when d.has_recent_sales and d.daily_demand_stddev_90d > 0 and sup.lead_time_days > 0
                then p.z_score * d.daily_demand_stddev_90d * sqrt(sup.lead_time_days)
            end,
            0
        ) as safety_stock,
        
        case
            when d.has_recent_sales and d.avg_daily_demand_90d > 0
            then ceil(
                (d.avg_daily_demand_90d * coalesce(sup.lead_time_days, 0))
                + case
                    when d.daily_demand_stddev_90d > 0 and sup.lead_time_days > 0
                    then p.z_score * d.daily_demand_stddev_90d * sqrt(sup.lead_time_days)
                    else 0
                end
            )
        end as calculated_reorder_point
        
    from stock s
    left join demand d 
        on s.product_variant_id = d.product_variant_id
        and s.location_id = d.location_id
    left join suppliers sup 
        on s.product_variant_id = sup.product_variant_id
    cross join params p
),

final as (
    select
        product_variant_id,
        location_id,
        current_stock_qty,
        avg_daily_demand_90d,
        daily_demand_stddev_90d,
        has_recent_sales,
        lead_time_days,
        lead_time_demand,
        safety_stock,
        source_reorder_point,
        calculated_reorder_point,
        
        coalesce(source_reorder_point, calculated_reorder_point) as effective_reorder_point,
        
        case
            when source_reorder_point is not null then 'source'
            when calculated_reorder_point is not null then 'calculated'
        end as reorder_point_origin
        
    from calculated
)

select * from final
