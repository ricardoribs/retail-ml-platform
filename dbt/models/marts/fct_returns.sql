with return_items as (

    select * from {{ ref('stg_return_items') }}

),

returns as (

    select * from {{ ref('stg_returns') }}

),

order_items as (

    select * from {{ ref('stg_order_items') }}

),

reason_classification as (

    select * from {{ ref('int_return_reason_classification') }}

),

final as (

    select
        {{ dbt_utils.generate_surrogate_key(['ri.return_item_id']) }} as returns_key,
        ri.return_item_id,
        r.return_id,
        r.order_id,
        oi.product_variant_id,
        cast(r.returned_at as date) as date_key,
        rc.return_reason_class,
        ri.return_action,
        ri.quantity_returned_qty,
        ri.unit_refund_amount,
        ri.return_amount
    from return_items ri
    inner join returns r
        on ri.return_id = r.return_id
    left join order_items oi
        on ri.order_item_id = oi.order_item_id
    left join reason_classification rc
        on r.return_id = rc.return_id

)

select * from final
