with payments as (

    select * from {{ ref('stg_payments') }}

),

orders as (

    select 
        order_id,
        cast(placed_at as date) as order_date,
        channel,
        order_status,
        customer_id
    from {{ ref('stg_orders') }}

),

payments_enriched as (

    select
        p.payment_id,
        p.order_id,
        case
            when o.customer_id is null then cast(-1 as string)
            else {{ dbt_utils.generate_surrogate_key(['o.customer_id']) }}
        end as customer_key,
        cast(p.paid_at as date) as payment_date,
        p.paid_at as payment_datetime,
        o.order_date,
        p.payment_method,
        p.payment_status,
        p.installments,
        p.amount as payment_amount_brl,
        o.channel as order_channel,
        o.order_status,
        datediff(day, o.order_date, cast(p.paid_at as date)) as days_from_order_to_payment,
        (p.payment_status = 'approved') as is_approved_flag,
        (p.payment_status = 'refunded') as is_refunded_flag,
        (p.payment_status = 'pending') as is_pending_flag,
        (p.payment_status = 'refused') as is_refused_flag,
        (p.payment_method = 'credit_card') as is_credit_card_flag,
        (p.payment_method = 'debit_card') as is_debit_card_flag,
        (p.payment_method = 'boleto') as is_boleto_flag,
        (p.payment_method = 'pix') as is_pix_flag,
        (p.installments > 1) as is_installment_flag,
        case
            when p.installments = 1 then 'upfront'
            when p.installments between 2 and 3 then '2-3x'
            when p.installments between 4 and 6 then '4-6x'
            when p.installments between 7 and 12 then '7-12x'
            when p.installments > 12 then '12x+'
            else 'not provided'
        end as installment_range
    from payments p
    left join orders o
        on p.order_id = o.order_id

)

select * from payments_enriched
