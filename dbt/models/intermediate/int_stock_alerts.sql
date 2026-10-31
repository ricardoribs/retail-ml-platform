/*
IMPORTANT: This model is retained for compatibility, but now uses the
effective reorder point (`effective_reorder_point`) calculated in
`int_reorder_point_calculated`, which applies COALESCE to the source value
and the value calculated by the safety-stock formula.

For complete replenishment analysis, use `fct_inventory_replenishment_snapshot`.
*/

with reorder_calc as (
    select
        product_variant_id,
        location_id,
        current_stock_qty as quantity_on_hand,
        source_reorder_point as reorder_point,
        effective_reorder_point,
        reorder_point_origin
    from {{ ref('int_reorder_point_calculated') }}
)

select
    product_variant_id,
    location_id,
    quantity_on_hand,
    reorder_point,
    effective_reorder_point,
    reorder_point_origin,
    case
        when quantity_on_hand <= effective_reorder_point then true
        else false
    end as is_reorder_needed,
    current_timestamp() as updated_at
from reorder_calc
where effective_reorder_point is not null
