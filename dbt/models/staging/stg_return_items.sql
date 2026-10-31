with source_data as (

    select * from {{ source('bronze', 'return_items') }}

),

cleaned_data as (

    select
        cast(id as bigint)                     as return_item_id,
        cast(return_id as bigint)              as return_id,
        cast(order_item_id as bigint)          as order_item_id,
        cast(quantity as decimal(18,3))        as quantity_returned_qty,
        cast(action as string)                 as return_action,
        cast(exchange_variant_id as bigint)    as exchange_variant_id,
        cast(unit_refund_amount as decimal(18,2)) as unit_refund_amount
    from source_data

),

deduped_data as (

    select
        *,
        row_number() over (
            partition by return_item_id
            order by return_item_id
        ) as rn
    from cleaned_data

),

final as (

    select
        return_item_id,
        return_id,
        order_item_id,
        quantity_returned_qty,
        return_action,
        exchange_variant_id,
        unit_refund_amount,
        quantity_returned_qty * unit_refund_amount as return_amount
    from deduped_data
    where rn = 1

)

select * from final
