/*
Custom test: no future record should be classified as "realized".
Model: int_daily_demand
Rule: sale_date must always be <= reference_date.
*/

with params as (
    select {{ var('forecast_config')['reference_date_sql'] }} as reference_date
)

select
    dd.sale_date,
    dd.product_variant_id,
    dd.location_id,
    dd.daily_demand_qty,
    p.reference_date
from {{ ref('int_daily_demand') }} dd
cross join params p
where dd.sale_date > p.reference_date
