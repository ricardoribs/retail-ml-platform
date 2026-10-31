with params as (
    select {{ var('forecast_config')['reference_date_sql'] }} as reference_date
),

received_items as (
    select
        gri.purchase_order_item_id,
        sum(gri.quantity_received) as total_received
    from {{ ref('stg_goods_receipt_items') }} gri
    join {{ ref('stg_goods_receipts') }} gr 
        on gri.goods_receipt_id = gr.goods_receipt_id
    group by gri.purchase_order_item_id
),

open_items as (
    select
        poi.product_variant_id,
        po.destination_location_id as location_id,
        poi.quantity_ordered,
        coalesce(ri.total_received, 0) as quantity_received,
        greatest(
            poi.quantity_ordered - coalesce(ri.total_received, 0),
            0
        ) as open_qty,
        po.expected_delivery_at,
        po.status
    from {{ ref('stg_purchase_order_items') }} poi
    join {{ ref('stg_purchase_orders') }} po 
        on poi.purchase_order_id = po.purchase_order_id
    left join received_items ri 
        on poi.purchase_order_item_id = ri.purchase_order_item_id
    where po.status in ({{ "'" + var('inventory_config')['valid_purchase_statuses'] | join("', '") + "'" }})
      and greatest(poi.quantity_ordered - coalesce(ri.total_received, 0), 0) > 0
),

aggregated as (
    select
        oi.product_variant_id,
        oi.location_id,
        
        sum(oi.open_qty) as open_purchase_qty,
        
        sum(case 
            when oi.expected_delivery_at < p.reference_date 
            then oi.open_qty 
            else 0 
        end) as overdue_purchase_qty,
        
        sum(case 
            when oi.expected_delivery_at >= p.reference_date 
            then oi.open_qty 
            else 0 
        end) as future_purchase_qty
        
    from open_items oi
    cross join params p
    group by 
        oi.product_variant_id,
        oi.location_id
)

select * from aggregated
