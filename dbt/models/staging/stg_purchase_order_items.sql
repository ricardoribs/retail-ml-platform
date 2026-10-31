with source_data as (

    select * from {{ source('bronze', 'purchase_order_items') }}

),

cleaned_data as (

    select
        cast(id as bigint)                    as purchase_order_item_id,
        cast(purchase_order_id as bigint)     as purchase_order_id,
        cast(product_variant_id as bigint)    as product_variant_id,
        cast(quantity_ordered as int)         as quantity_ordered,
        cast(unit_cost as decimal(18,2))      as unit_cost,
        cast(line_total as decimal(18,2))     as line_total
    from source_data

),

deduped_data as (

    -- Remove duplicates, keeping one row per purchase_order_item_id.
    select
        *,
        row_number() over (
            partition by purchase_order_item_id
            order by purchase_order_item_id
        ) as rn
    from cleaned_data

),

final as (

    select
        purchase_order_item_id,
        purchase_order_id,
        product_variant_id,
        quantity_ordered,
        unit_cost,
        line_total
    from deduped_data
    where rn = 1

)

select * from final
