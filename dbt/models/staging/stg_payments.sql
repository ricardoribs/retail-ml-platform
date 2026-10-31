with source_data as (

    select * from {{ source('bronze', 'payments') }}

),

cleaned_data as (

    select
        cast(id as bigint)              as payment_id,
        cast(order_id as bigint)        as order_id,
        cast(method as string)          as payment_method,
        cast(installments as int)       as installments,
        cast(amount as decimal(18,2))   as amount,
        cast(status as string)          as payment_status,
        cast(paid_at as timestamp)      as paid_at
    from source_data

),

deduped_data as (

    select
        *,
        row_number() over (
            partition by payment_id
            order by paid_at desc
        ) as rn
    from cleaned_data

),

final as (

    select
        payment_id,
        order_id,
        payment_method,
        installments,
        amount,
        payment_status,
        paid_at
    from deduped_data
    where rn = 1

)

select * from final
