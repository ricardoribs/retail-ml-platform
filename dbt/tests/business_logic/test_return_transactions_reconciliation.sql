select
    return_id,
    total_refund_amount,
    total_items_refund_amount,
    refund_reconciliation_difference
from {{ ref('fct_return_transactions') }}
where abs(coalesce(refund_reconciliation_difference, 0)) > 0.01