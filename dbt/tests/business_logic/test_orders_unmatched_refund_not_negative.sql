select *
from {{ ref('fct_orders') }}
where unmatched_refund_amount_brl < 0
