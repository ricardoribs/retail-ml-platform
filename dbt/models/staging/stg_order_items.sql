with source_data as (

    select * from {{ source('bronze', 'order_items') }}

),

cleaned_data as (

    select
        cast(id as bigint)                    as order_item_id,
        cast(order_id as bigint)              as order_id,
        cast(product_variant_id as bigint)    as product_variant_id,
        cast(quantity as decimal(18,3))       as quantity_qty,
        cast(unit_price as decimal(18,2))     as unit_price,
        cast(icms_rate as decimal(5,2))       as icms_rate,
        cast(ipi_rate as decimal(5,2))        as ipi_rate,
        cast(line_total as decimal(18,2))     as line_total
    from source_data

),

deduped_data as (

    select
        *,
        row_number() over (
            partition by order_item_id
            order by order_item_id
        ) as rn
    from cleaned_data

),

final as (

    select
        order_item_id,
        order_id,
        product_variant_id,
        quantity_qty,
        unit_price,
        icms_rate,
        ipi_rate,
        line_total
    from deduped_data
    where rn = 1

)

select * from final
