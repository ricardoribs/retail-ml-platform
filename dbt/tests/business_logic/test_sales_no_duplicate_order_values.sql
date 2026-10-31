with order_totals_in_sales as (

    select
        order_id,
        sum(line_total) as total_line_from_sales,
        count(*) as item_count
    from {{ ref('fct_sales') }}
    group by order_id

),

order_totals_in_items as (

    select
        order_id,
        sum(line_total) as total_line_from_items,
        count(*) as item_count
    from {{ ref('stg_order_items') }}
    group by order_id

),

comparison as (

    select
        s.order_id,
        s.total_line_from_sales,
        i.total_line_from_items,
        s.item_count as sales_item_count,
        i.item_count as source_item_count,
        abs(s.total_line_from_sales - i.total_line_from_items) as value_difference
    from order_totals_in_sales s
    inner join order_totals_in_items i
        on s.order_id = i.order_id

)

select *
from comparison
where value_difference > 0.01
   or sales_item_count != source_item_count
