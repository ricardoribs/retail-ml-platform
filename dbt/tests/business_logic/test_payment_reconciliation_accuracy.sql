select
    order_id,
    total_paid_amount_brl,
    total_refunded_amount_brl,
    raw_net_revenue_amount_brl,
    (total_paid_amount_brl - total_refunded_amount_brl) as calculated_net_revenue,
    abs(raw_net_revenue_amount_brl - (total_paid_amount_brl - total_refunded_amount_brl)) as difference
from {{ ref('int_payment_reconciliation') }}
where abs(raw_net_revenue_amount_brl - (total_paid_amount_brl - total_refunded_amount_brl)) > 0.01
