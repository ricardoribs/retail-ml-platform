with params as (
    select {{ forecast_cutoff_date() }} as cutoff_date
)

select demand.*
from {{ ref('int_monthly_product_location_demand') }} demand
cross join params p
where demand.demand_month > p.cutoff_date
