/*
Custom test: alerts without source_reorder_point use calculated_reorder_point.
Model: fct_inventory_replenishment_snapshot
Rule: When source_reorder_point is NULL, effective_reorder_point must equal calculated_reorder_point.
Goal: Validate that COALESCE works as expected.
*/

select 
    product_variant_id,
    location_id,
    source_reorder_point,
    calculated_reorder_point,
    effective_reorder_point
from {{ ref('fct_inventory_replenishment_snapshot') }}
where source_reorder_point is null
  and effective_reorder_point is not null
  and effective_reorder_point != calculated_reorder_point
