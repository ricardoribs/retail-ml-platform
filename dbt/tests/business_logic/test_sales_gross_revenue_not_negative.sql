select *
from {{ ref('fct_sales') }}
where gross_revenue_brl < 0
