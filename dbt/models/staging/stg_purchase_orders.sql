with source_data as (

    select * from {{ source('bronze', 'purchase_orders') }}

),

cleaned_data as (

    select
        cast(id as bigint)                        as purchase_order_id,
        cast(po_number as string)                 as po_number,
        cast(supplier_id as bigint)               as supplier_id,
        cast(buyer_id as bigint)                  as buyer_id,
        cast(destination_location_id as bigint)   as destination_location_id,
        cast(status as string)                    as status,
        cast(currency as string)                  as currency,
        cast(subtotal as decimal(18,2))           as subtotal,
        cast(total as decimal(18,2))              as total_amount,
        cast(placed_at as timestamp)              as placed_at,
        cast(expected_delivery_at as date)        as expected_delivery_at,
        cast(created_at as timestamp)             as created_at,
        cast(updated_at as timestamp)             as updated_at
    from source_data

),

deduped_data as (

    -- Remove duplicates, keeping one row per purchase_order_id.
    select
        *,
        row_number() over (
            partition by purchase_order_id
            order by updated_at desc
        ) as rn
    from cleaned_data

),

final as (

    select
        purchase_order_id,
        po_number,
        supplier_id,
        buyer_id,
        destination_location_id,
        status,
        currency,
        subtotal,
        total_amount,
        placed_at,
        expected_delivery_at,
        created_at,
        updated_at
    from deduped_data
    where rn = 1

)

select * from final
