/*
Custom test: products without recent sales must not have a suggested purchase.
Modelo: fct_inventory_replenishment_snapshot
Rule: has_recent_sales = FALSE => suggested_purchase_qty = 0
*/

select
    reference_date,
    product_variant_id,
    location_id,
    has_recent_sales,
    suggested_purchase_qty
from {{ ref('fct_inventory_replenishment_snapshot') }}
where has_recent_sales = false
  and suggested_purchase_qty > 0
