with source_data as (

    select * from {{ source('bronze', 'stock_levels') }}

),

cleaned_data as (

    select
        cast(product_variant_id as bigint)      as product_variant_id,
        cast(location_id as bigint)              as location_id,
        cast(quantity_on_hand as decimal(18,3))  as quantity_on_hand,
        cast(reorder_point as decimal(18,3))     as reorder_point,
        cast(updated_at as timestamp)            as updated_at
    from source_data

),

deduped_data as (

    -- Remove duplicates, keeping one row per (product_variant_id, location_id).
    select
        *,
        row_number() over (
            partition by product_variant_id, location_id
            order by updated_at desc
        ) as rn
    from cleaned_data

),

final as (

    select
        product_variant_id,
        location_id,
        quantity_on_hand,
        reorder_point,
        updated_at
    from deduped_data
    where rn = 1

)

select * from final
