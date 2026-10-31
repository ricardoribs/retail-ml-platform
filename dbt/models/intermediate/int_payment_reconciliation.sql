with payments as (

    select * from {{ ref('stg_payments') }}

),

payments_by_status as (

    select
        order_id,
        sum(case when payment_status = 'approved' then amount else 0 end) as total_paid_amount_brl,
        sum(case when payment_status = 'refunded' then amount else 0 end) as total_refunded_amount_brl,
        sum(case when payment_status = 'pending' then amount else 0 end) as total_pending_amount_brl,
        sum(case when payment_status = 'refused' then amount else 0 end) as total_refused_amount_brl,
        count(case when payment_status = 'approved' then 1 end) as approved_payments_count,
        count(case when payment_status = 'refunded' then 1 end) as refunded_payments_count,
        count(case when payment_status = 'pending' then 1 end) as pending_payments_count,
        count(case when payment_status = 'refused' then 1 end) as refused_payments_count,
        count(*) as total_payments_count
    from payments
    group by order_id

),

reconciled as (

    select
        order_id,
        total_paid_amount_brl,
        total_refunded_amount_brl,
        total_pending_amount_brl,
        total_refused_amount_brl,
        approved_payments_count,
        refunded_payments_count,
        pending_payments_count,
        refused_payments_count,
        total_payments_count,
        (total_paid_amount_brl - total_refunded_amount_brl) as raw_net_revenue_amount_brl,
        case
            when total_refunded_amount_brl > total_paid_amount_brl 
            then total_refunded_amount_brl - total_paid_amount_brl
            else 0
        end as unmatched_refund_amount_brl,
        case
            when (total_paid_amount_brl - total_refunded_amount_brl) < 0 then 0
            else (total_paid_amount_brl - total_refunded_amount_brl)
        end as recognized_net_revenue_amount_brl,
        case
            when total_refunded_amount_brl > total_paid_amount_brl 
            then total_paid_amount_brl
            else total_refunded_amount_brl
        end as recognized_refund_amount_brl,
        total_pending_amount_brl as pending_potential_value_brl,
        (approved_payments_count > 0) as has_approved_payment_flag,
        (refunded_payments_count > 0) as has_refund_flag,
        (pending_payments_count > 0) as has_pending_payment_flag,
        (refused_payments_count > 0) as has_refused_payment_flag,
        (refunded_payments_count > 0 and approved_payments_count = 0) as has_refund_without_approved_payment_flag,
        (total_refunded_amount_brl > total_paid_amount_brl) as has_refund_exceeding_paid_flag,
        case
            when refunded_payments_count > 0 and approved_payments_count = 0 
            then 'refund_without_approved_payment'
            when total_refunded_amount_brl > total_paid_amount_brl 
            then 'refund_exceeds_paid'
            when approved_payments_count > 0 
                and (pending_payments_count > 0 or refused_payments_count > 0)
            then 'partial_payment'
            when pending_payments_count > 0 and approved_payments_count = 0
            then 'pending_only'
            when refused_payments_count > 0 and approved_payments_count = 0
            then 'refused_only'
            when total_payments_count = 0
            then 'no_financial_movement'
            else 'reconciled'
        end as payment_reconciliation_status
    from payments_by_status

)

select * from reconciled
