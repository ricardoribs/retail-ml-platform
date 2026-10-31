with orders_values as (

    select
        order_id,
        total_paid_amount_brl,
        total_refunded_amount_brl,
        raw_net_revenue_amount_brl,
        recognized_net_revenue_amount_brl,
        unmatched_refund_amount_brl
    from {{ ref('fct_orders') }}

),

reconciliation_values as (

    select
        order_id,
        total_paid_amount_brl,
        total_refunded_amount_brl,
        raw_net_revenue_amount_brl,
        recognized_net_revenue_amount_brl,
        unmatched_refund_amount_brl
    from {{ ref('int_payment_reconciliation') }}

),

comparison as (

    select
        o.order_id,
        abs(coalesce(o.total_paid_amount_brl, 0) - coalesce(r.total_paid_amount_brl, 0)) as diff_paid,
        abs(coalesce(o.total_refunded_amount_brl, 0) - coalesce(r.total_refunded_amount_brl, 0)) as diff_refunded,
        abs(coalesce(o.raw_net_revenue_amount_brl, 0) - coalesce(r.raw_net_revenue_amount_brl, 0)) as diff_raw_revenue,
        abs(coalesce(o.recognized_net_revenue_amount_brl, 0) - coalesce(r.recognized_net_revenue_amount_brl, 0)) as diff_recognized_revenue,
        abs(coalesce(o.unmatched_refund_amount_brl, 0) - coalesce(r.unmatched_refund_amount_brl, 0)) as diff_unmatched_refund
    from orders_values o
    inner join reconciliation_values r
        on o.order_id = r.order_id

)

select *
from comparison
where diff_paid > 0.01
   or diff_refunded > 0.01
   or diff_raw_revenue > 0.01
   or diff_recognized_revenue > 0.01
   or diff_unmatched_refund > 0.01
