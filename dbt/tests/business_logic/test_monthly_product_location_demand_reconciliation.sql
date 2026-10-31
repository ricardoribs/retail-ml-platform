with params as (

    select {{ forecast_cutoff_date() }} as cutoff_date

),

expected as (

    select
        trunc(cast(s.sale_datetime as date), 'month') as demand_month,
        s.product_variant_id,
        l.location_id,
        sum(s.quantity_qty) as demand_qty,
        sum(s.gross_revenue_brl) as gross_revenue_brl,
        count(distinct s.order_id) as order_count,
        count(*) as item_line_count,
        count(
            distinct case
                when s.quantity_qty > 0 then cast(s.sale_datetime as date)
            end
        ) as days_with_sales
    from {{ ref('fct_sales') }} s
    inner join {{ ref('dim_location') }} l
        on s.location_key = l.location_key
    inner join {{ ref('dim_product') }} p
        on s.product_variant_id = p.product_variant_id
    cross join params pms
    where s.order_status = 'paid'
      and cast(s.sale_datetime as date) <= pms.cutoff_date
    group by
        trunc(cast(s.sale_datetime as date), 'month'),
        s.product_variant_id,
        l.location_id

),

actual as (

    select
        demand_month,
        product_variant_id,
        location_id,
        demand_qty,
        gross_revenue_brl,
        order_count,
        item_line_count,
        days_with_sales
    from {{ ref('int_monthly_product_location_demand') }}
    where item_line_count > 0

),

comparison as (

    select
        coalesce(e.demand_month, a.demand_month) as demand_month,
        coalesce(e.product_variant_id, a.product_variant_id) as product_variant_id,
        coalesce(e.location_id, a.location_id) as location_id,
        e.demand_qty as expected_demand_qty,
        a.demand_qty as actual_demand_qty,
        e.gross_revenue_brl as expected_gross_revenue_brl,
        a.gross_revenue_brl as actual_gross_revenue_brl,
        e.order_count as expected_order_count,
        a.order_count as actual_order_count,
        e.item_line_count as expected_item_line_count,
        a.item_line_count as actual_item_line_count,
        e.days_with_sales as expected_days_with_sales,
        a.days_with_sales as actual_days_with_sales
    from expected e
    full outer join actual a
        on e.demand_month = a.demand_month
        and e.product_variant_id = a.product_variant_id
        and e.location_id = a.location_id

)

select *
from comparison
where expected_demand_qty is null
   or actual_demand_qty is null
   or abs(expected_demand_qty - actual_demand_qty) > 0.000001
   or abs(expected_gross_revenue_brl - actual_gross_revenue_brl) > 0.01
   or expected_order_count != actual_order_count
   or expected_item_line_count != actual_item_line_count
   or expected_days_with_sales != actual_days_with_sales
