with order_items as (

    select * from {{ ref('stg_order_items') }}

),

orders as (

    select * from {{ ref('stg_orders') }}

),

products as (

    select * from {{ ref('dim_product') }}

),

dates as (

    select * from {{ ref('dim_date') }}

),

sales_enriched as (

    select
        {{ dbt_utils.generate_surrogate_key(['oi.order_item_id']) }} as sales_key,
        oi.order_item_id,
        oi.order_id,
        {{ dbt_utils.generate_surrogate_key(['o.customer_id']) }} as customer_key,
        p.product_key,
        {{ dbt_utils.generate_surrogate_key(['o.location_id']) }} as location_key,
        case
            when o.salesperson_id is null then cast(-1 as string)
            else {{ dbt_utils.generate_surrogate_key(['o.salesperson_id']) }}
        end as employee_key,
        d.date_key,
        o.placed_at as sale_datetime,
        o.channel,
        o.order_status,
        oi.product_variant_id,
        oi.quantity_qty,
        oi.unit_price,
        coalesce(oi.icms_rate, 0) as icms_rate,
        coalesce(oi.ipi_rate, 0) as ipi_rate,
        oi.line_total,
        oi.line_total - (oi.line_total * (coalesce(oi.icms_rate, 0) + coalesce(oi.ipi_rate, 0)) / 100) as gross_revenue_brl,
        (oi.line_total * (coalesce(oi.icms_rate, 0) + coalesce(oi.ipi_rate, 0)) / 100) as tax_amount_brl
    from order_items oi
    inner join orders o
        on oi.order_id = o.order_id
    left join products p
        on oi.product_variant_id = p.product_variant_id
    left join dates d
        on cast(o.placed_at as date) = d.full_date

)

select * from sales_enriched
