with source_data as (

    select * from {{ source('bronze', 'returns') }}

),

cleaned_data as (

    select
        cast(id as bigint)                       as return_id,
        cast(return_number as string)             as return_number,
        cast(order_id as bigint)                  as order_id,
        cast(customer_id as bigint)               as customer_id,
        cast(received_at_location_id as bigint)   as received_at_location_id,
        cast(status as string)                    as return_status,
        cast(reason as string)                    as return_reason,
        cast(total_refund_amount as decimal(18,2)) as total_refund_amount,
        cast(created_at as timestamp)             as returned_at
    from source_data

),

deduped_data as (

    select
        *,
        row_number() over (
            partition by return_id
            order by returned_at desc
        ) as rn
    from cleaned_data

),

final as (

    select
        return_id,
        return_number,
        order_id,
        customer_id,
        received_at_location_id,
        return_status,
        return_reason,
        total_refund_amount,
        returned_at
    from deduped_data
    where rn = 1

)

select * from final
