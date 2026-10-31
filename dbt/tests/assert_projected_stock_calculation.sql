/*
Custom test: validate the projected inventory calculation.
Modelo: fct_inventory_replenishment_snapshot
Rule: projected_stock_qty must always equal current_stock_qty + open_purchase_qty.
*/

select
    reference_date,
    product_variant_id,
    location_id,
    current_stock_qty,
    open_purchase_qty,
    projected_stock_qty,
    (current_stock_qty + open_purchase_qty) as expected_projected_stock
from {{ ref('fct_inventory_replenishment_snapshot') }}
where projected_stock_qty != (current_stock_qty + open_purchase_qty)
