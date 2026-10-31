with source_data as (

    select * from {{ source('bronze', 'goods_receipts') }}

),

cleaned_data as (

    select
        cast(id as bigint)                      as goods_receipt_id,
        cast(purchase_order_id as bigint)       as purchase_order_id,
        cast(received_by_employee_id as bigint) as received_by_employee_id,
        cast(received_at as timestamp)          as received_at,
        cast(notes as string)                   as notes,
        cast(created_at as timestamp)           as created_at
    from source_data

),

deduped_data as (

    -- Remove duplicates, keeping one row per goods_receipt_id.
    select
        *,
        row_number() over (
            partition by goods_receipt_id
            order by created_at desc
        ) as rn
    from cleaned_data

),

final as (

    select
        goods_receipt_id,
        purchase_order_id,
        received_by_employee_id,
        received_at,
        notes,
        created_at
    from deduped_data
    where rn = 1

)

select * from final
