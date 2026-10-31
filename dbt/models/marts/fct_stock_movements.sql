with stock_movements as (

    select * from {{ ref('stg_stock_movements') }}

),

dim_date as (

    select * from {{ ref('dim_date') }}

),

dim_product as (

    select * from {{ ref('dim_product') }}

),

dim_location as (

    select * from {{ ref('dim_location') }}

),

movements_enriched as (

    select
        {{ dbt_utils.generate_surrogate_key(['sm.stock_movement_id']) }} as stock_movement_key,
        sm.stock_movement_id,
        dp.product_key,
        dl.location_key,
        dd.date_key,
        sm.movement_type,
        sm.quantity as quantity_moved_qty,
        sm.reference_table,
        sm.reference_id,
        sm.employee_id,
        sm.notes,
        sm.occurred_at,
        sm.created_at
    from stock_movements sm
    left join dim_product dp
        on sm.product_variant_id = dp.product_variant_id
    left join dim_location dl
        on sm.location_id = dl.location_id
    left join dim_date dd
        on cast(sm.occurred_at as date) = dd.full_date

)

select * from movements_enriched
