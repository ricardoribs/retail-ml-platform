with source_data as (

    -- Select raw data from the Bronze "orders" table.
    select * from {{ source('bronze', 'orders') }}

),

cleaned_data as (

    select
        cast(id as bigint)                    as order_id,
        cast(order_number as string)          as order_number,
        cast(channel as string)               as channel,
        cast(customer_id as bigint)           as customer_id,
        cast(salesperson_id as bigint)        as salesperson_id,
        cast(location_id as bigint)           as location_id,
        cast(status as string)                as order_status,
        cast(subtotal as decimal(18,2))       as subtotal,
        cast(discount_amount as decimal(18,2)) as discount_amount,
        cast(total as decimal(18,2))          as total_amount,
        cast(placed_at as timestamp)          as placed_at
    from source_data

),

deduped_data as (

    -- Remove duplicates, keeping one row per order_id.
    select
        *,
        row_number() over (
            partition by order_id
            order by placed_at desc
        ) as rn
    from cleaned_data

),

final as (

    select
        order_id,
        order_number,
        channel,
        customer_id,
        salesperson_id,
        location_id,
        order_status,
        subtotal,
        discount_amount,
        total_amount,
        placed_at
    from deduped_data
    where rn = 1

)

select * from final
