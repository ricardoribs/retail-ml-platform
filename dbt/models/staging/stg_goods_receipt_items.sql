with source_data as (

    select * from {{ source('bronze', 'goods_receipt_items') }}

),

cleaned_data as (

    select
        cast(id as bigint)                     as goods_receipt_item_id,
        cast(goods_receipt_id as bigint)       as goods_receipt_id,
        cast(purchase_order_item_id as bigint) as purchase_order_item_id,
        cast(quantity_received as int)         as quantity_received
    from source_data

),

deduped_data as (

    -- Remove duplicates, keeping one row per goods_receipt_item_id.
    select
        *,
        row_number() over (
            partition by goods_receipt_item_id
            order by goods_receipt_item_id
        ) as rn
    from cleaned_data

),

final as (

    select
        goods_receipt_item_id,
        goods_receipt_id,
        purchase_order_item_id,
        quantity_received
    from deduped_data
    where rn = 1

)

select * from final
