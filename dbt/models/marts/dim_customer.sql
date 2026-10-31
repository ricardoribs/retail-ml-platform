with customers as (

    select * from {{ ref('stg_customers') }}

),

addresses as (

    select * from {{ ref('stg_addresses') }}
    where is_primary_flag = true

),

-- Detects whether the customer has purchased through both the physical store (POS) and e-commerce.
customer_channels as (

    select
        customer_id,
        max(case when channel = 'pos' then 1 else 0 end) as has_instore_purchase,
        max(case when channel = 'ecommerce' then 1 else 0 end) as has_ecommerce_purchase
    from {{ ref('stg_orders') }}
    group by customer_id

),

joined as (

    select
        c.customer_id,
        c.person_type,
        c.customer_name,
        c.trade_name,
        -- Hash the email to avoid exposing personal data in plain text (LGPD).
        sha2(c.email, 256) as email_hash,
        a.city,
        a.state,
        a.country,
        c.is_active_flag,
        coalesce(ch.has_instore_purchase, 0) as has_instore_purchase,
        coalesce(ch.has_ecommerce_purchase, 0) as has_ecommerce_purchase,
        case
            when coalesce(ch.has_instore_purchase, 0) = 1
             and coalesce(ch.has_ecommerce_purchase, 0) = 1
            then true else false
        end as is_omnichannel
    from customers c
    left join addresses a
        on c.customer_id = a.customer_id
    left join customer_channels ch
        on c.customer_id = ch.customer_id

),

final as (

    select
        {{ dbt_utils.generate_surrogate_key(['customer_id']) }} as customer_key,
        customer_id,
        person_type,
        customer_name,
        trade_name,
        email_hash,
        city,
        state,
        country,
        is_active_flag,
        has_instore_purchase,
        has_ecommerce_purchase,
        is_omnichannel
    from joined

)

select * from final
