with goods_receipt_items as (

    select * from {{ ref('stg_goods_receipt_items') }}

),

goods_receipts as (

    select * from {{ ref('stg_goods_receipts') }}

),

purchase_order_items as (

    select * from {{ ref('stg_purchase_order_items') }}

),

purchase_orders as (

    select * from {{ ref('stg_purchase_orders') }}

),

dim_product as (

    select * from {{ ref('dim_product') }}

),

dim_location as (

    select * from {{ ref('dim_location') }}

),

dim_date as (

    select * from {{ ref('dim_date') }}

),

dim_supplier as (

    select * from {{ ref('dim_supplier') }}

),

purchases_enriched as (

    select
        {{ dbt_utils.generate_surrogate_key(['gri.goods_receipt_item_id']) }} as purchase_key,
        gri.goods_receipt_item_id,
        gr.goods_receipt_id,
        poi.purchase_order_id,
        po.supplier_id,
        ds.supplier_key,
        dp.product_key,
        dl.location_key,
        dd.date_key,
        po.status as purchase_order_status,
        gri.quantity_received as quantity_received_qty,
        poi.unit_cost as unit_cost_brl,
        gri.quantity_received * poi.unit_cost as total_cost_brl,
        gr.received_by_employee_id,
        gr.received_at
    from goods_receipt_items gri
    inner join goods_receipts gr
        on gri.goods_receipt_id = gr.goods_receipt_id
    inner join purchase_order_items poi
        on gri.purchase_order_item_id = poi.purchase_order_item_id
    inner join purchase_orders po
        on poi.purchase_order_id = po.purchase_order_id
    left join dim_product dp
        on poi.product_variant_id = dp.product_variant_id
    left join dim_location dl
        on po.destination_location_id = dl.location_id
    left join dim_date dd
        on cast(gr.received_at as date) = dd.full_date
    left join dim_supplier ds
        on po.supplier_id = ds.supplier_id

)

select * from purchases_enriched
