{{
    config(
        materialized='incremental',
        unique_key=['reference_date', 'product_variant_id', 'location_id'],
        incremental_strategy='merge',
        schema='marts',
        tags=['inventory', 'snapshot']
    )
}}

with params as (
    select
        {{ var('forecast_config')['reference_date_sql'] }} as reference_date,
        {{ var('inventory_config')['target_coverage_days'] }} as target_coverage_days,
        {{ var('inventory_config')['service_level_z_score'] }} as z_score
),

demand_metrics as (
    select * from {{ ref('int_inventory_demand_metrics') }}
),

reorder_calc as (
    select * from {{ ref('int_reorder_point_calculated') }}
),

open_purchases as (
    select * from {{ ref('int_open_purchases') }}
),

preferred_suppliers as (
    select * from {{ ref('int_preferred_suppliers') }}
),

date_dim as (
    select 
        full_date,
        date_key
    from {{ ref('dim_date') }}
),

products as (
    select 
        product_variant_id,
        product_id,
        product_key
    from {{ ref('dim_product') }}
),

locations as (
    select 
        location_id,
        location_key
    from {{ ref('dim_location') }}
),

suppliers as (
    select
        supplier_id,
        supplier_key
    from {{ ref('dim_supplier') }}
),

-- Consolidate all data.
consolidated as (
    select
        p.reference_date,
        
        -- Dimension keys for Power BI.
        dd.date_key,
        rc.product_variant_id,
        prd.product_key,
        rc.location_id,
        loc.location_key,
        ps.supplier_id,
        sup.supplier_key,
        
        -- Demand metrics.
        coalesce(dm.demand_30d_qty, 0) as demand_30d_qty,
        coalesce(dm.demand_60d_qty, 0) as demand_60d_qty,
        coalesce(dm.demand_90d_qty, 0) as demand_90d_qty,
        coalesce(dm.avg_daily_demand_30d, 0) as avg_daily_demand_30d,
        coalesce(dm.avg_daily_demand_60d, 0) as avg_daily_demand_60d,
        coalesce(dm.avg_daily_demand_90d, 0) as avg_daily_demand_90d,
        coalesce(dm.daily_demand_stddev_90d, 0) as daily_demand_stddev_90d,
        coalesce(dm.days_with_sales_90d, 0) as days_with_sales_90d,
        coalesce(dm.has_recent_sales, false) as has_recent_sales,
        
        -- Fornecimento
        coalesce(ps.lead_time_days, 0) as lead_time_days,
        coalesce(ps.last_quoted_cost, 0) as last_quoted_cost,
        
        -- Inventory.
        coalesce(rc.current_stock_qty, 0) as current_stock_qty,
        coalesce(op.open_purchase_qty, 0) as open_purchase_qty,
        coalesce(op.overdue_purchase_qty, 0) as overdue_purchase_qty,
        coalesce(op.future_purchase_qty, 0) as future_purchase_qty,
        coalesce(rc.current_stock_qty, 0) + coalesce(op.open_purchase_qty, 0) as projected_stock_qty,
        
        -- Reorder point.
        rc.lead_time_demand,
        rc.safety_stock,
        rc.source_reorder_point,
        rc.calculated_reorder_point,
        rc.effective_reorder_point,
        rc.reorder_point_origin,
        
        -- Indicadores operacionais
        case
            when coalesce(dm.avg_daily_demand_90d, 0) > 0
            then coalesce(rc.current_stock_qty, 0) / dm.avg_daily_demand_90d
        end as stock_coverage_days,
        
        case
            when rc.effective_reorder_point is not null 
                 and rc.current_stock_qty <= rc.effective_reorder_point
            then true
            else false
        end as is_reorder_needed,
        
        case
            when rc.effective_reorder_point is not null
                 and (rc.current_stock_qty + coalesce(op.open_purchase_qty, 0)) <= rc.effective_reorder_point
            then true
            else false
        end as is_reorder_needed_after_inbound,
        
        case
            when coalesce(dm.avg_daily_demand_90d, 0) > 0
                 and (coalesce(rc.current_stock_qty, 0) / dm.avg_daily_demand_90d) < coalesce(ps.lead_time_days, 0)
            then true
            else false
        end as is_stockout_risk,
        
        -- Flags de produtos especiais
        case
            when coalesce(dm.demand_90d_qty, 0) = 0 then true
            else false
        end as is_new_product,
        
        case
            when coalesce(dm.has_recent_sales, false) = false then true
            else false
        end as is_stale_product,
        
        -- Suggested purchase (30 additional days of coverage).
        case
            -- Do not suggest purchases for products without sales.
            when coalesce(dm.has_recent_sales, false) = false then 0
            when coalesce(dm.avg_daily_demand_90d, 0) = 0 then 0
            else greatest(
                (
                    -- Target: lead_time + 30 dias de cobertura + safety stock
                    (dm.avg_daily_demand_90d * (coalesce(ps.lead_time_days, 0) + p.target_coverage_days))
                    + coalesce(rc.safety_stock, 0)
                    - coalesce(rc.current_stock_qty, 0)
                    - coalesce(op.open_purchase_qty, 0)
                ),
                0
            )
        end as suggested_purchase_qty
        
    from params p
    cross join reorder_calc rc
    inner join date_dim dd
        on p.reference_date = dd.full_date
    left join demand_metrics dm
        on rc.product_variant_id = dm.product_variant_id
        and rc.location_id = dm.location_id
    left join open_purchases op
        on rc.product_variant_id = op.product_variant_id
        and rc.location_id = op.location_id
    left join preferred_suppliers ps
        on rc.product_variant_id = ps.product_variant_id
    inner join products prd
        on rc.product_variant_id = prd.product_variant_id
    inner join locations loc
        on rc.location_id = loc.location_id
    left join suppliers sup
        on ps.supplier_id = sup.supplier_id
),

final as (
    select
        *,
        
        -- Estimated cost of the suggested purchase.
        suggested_purchase_qty * last_quoted_cost as estimated_purchase_cost,
        
        -- Metadados
        current_timestamp() as created_at,
        'v1.0' as model_version
        
    from consolidated
)

select * from final

{% if is_incremental() %}
    where reference_date >= (select max(this.reference_date) from {{ this }} this)
{% endif %}
