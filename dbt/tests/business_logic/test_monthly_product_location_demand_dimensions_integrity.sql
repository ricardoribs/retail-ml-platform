with params as (

    select {{ forecast_cutoff_date() }} as cutoff_date

),

sales_scope as (

    select
        s.order_id,
        s.product_variant_id,
        s.location_key
    from {{ ref('fct_sales') }} s
    cross join params p
    where s.order_status = 'paid'
      and cast(s.sale_datetime as date) <= p.cutoff_date

),

missing_locations as (

    select
        'missing_location_dimension' as failure_type,
        cast(s.order_id as string) as order_id,
        cast(s.location_key as string) as missing_key
    from sales_scope s
    left join {{ ref('dim_location') }} l
        on s.location_key = l.location_key
    where l.location_key is null

),

missing_products as (

    select
        'missing_product_dimension' as failure_type,
        cast(s.order_id as string) as order_id,
        cast(s.product_variant_id as string) as missing_key
    from sales_scope s
    left join {{ ref('dim_product') }} p
        on s.product_variant_id = p.product_variant_id
    where p.product_variant_id is null

)

select * from missing_locations
union all
select * from missing_products
