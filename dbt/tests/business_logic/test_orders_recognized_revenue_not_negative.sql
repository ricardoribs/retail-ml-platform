select *
from {{ ref('fct_orders') }}
where recognized_net_revenue_amount_brl < 0
