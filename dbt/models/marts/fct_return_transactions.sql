{{
    config(
        materialized = 'table',
        schema = 'marts',
        tags = ['returns', 'finance']
    )
}}

with returns as (

    select *
    from {{ ref('stg_returns') }}

),

orders as (

    select *
    from {{ ref('stg_orders') }}

),

return_items as (

    select *
    from {{ ref('stg_return_items') }}

),

reason_classification as (

    select *
    from {{ ref('int_return_reason_classification') }}

),

dates as (

    select
        full_date,
        date_key
    from {{ ref('dim_date') }}

),

customers as (

    select
        customer_id,
        customer_key
    from {{ ref('dim_customer') }}

),

order_locations as (

    select
        location_id,
        location_key
    from {{ ref('dim_location') }}

),

received_locations as (

    select
        location_id,
        location_key
    from {{ ref('dim_location') }}

),

return_items_aggregated as (

    select
        return_id,

        count(distinct return_item_id)
            as return_items_count,

        sum(quantity_returned_qty)
            as total_quantity_returned_qty,

        sum(return_amount)
            as total_items_refund_amount,

        max(
            case
                when return_action = 'refund' then 1
                else 0
            end
        ) = 1 as has_refund_action,

        max(
            case
                when return_action = 'exchange' then 1
                else 0
            end
        ) = 1 as has_exchange_action

    from return_items
    group by return_id

),

final as (

    select
        {{
            dbt_utils.generate_surrogate_key(
                ['r.return_id']
            )
        }} as return_key,

        -- Identification.
        r.return_id,
        r.return_number,
        r.order_id,
        o.order_number,

        -- Dimension keys.
        c.customer_key,
        ol.location_key,
        rl.location_key as received_at_location_key,
        d.date_key,

        -- Dates.
        cast(r.returned_at as date) as returned_date,
        r.returned_at,

        -- Order context.
        cast(o.placed_at as date) as order_date,
        o.channel as order_channel,
        o.order_status,

        -- Return data.
        r.return_status,
        r.return_reason,
        rc.return_reason_class,
        r.total_refund_amount,

        -- Item totals.
        coalesce(ria.return_items_count, 0)
            as return_items_count,

        coalesce(ria.total_quantity_returned_qty, 0)
            as total_quantity_returned_qty,

        coalesce(ria.total_items_refund_amount, 0)
            as total_items_refund_amount,

        -- Reconciliation.
        r.total_refund_amount
            - coalesce(ria.total_items_refund_amount, 0)
            as refund_reconciliation_difference,

        -- Action types.
        coalesce(ria.has_refund_action, false)
            as has_refund_action,

        coalesce(ria.has_exchange_action, false)
            as has_exchange_action,

        (
            coalesce(ria.has_refund_action, false)
            and coalesce(ria.has_exchange_action, false)
        ) as has_mixed_actions

    from returns r

    left join orders o
        on r.order_id = o.order_id

    left join reason_classification rc
        on r.return_id = rc.return_id

    left join return_items_aggregated ria
        on r.return_id = ria.return_id

    left join dates d
        on cast(r.returned_at as date) = d.full_date

    left join customers c
        on r.customer_id = c.customer_id

    left join order_locations ol
        on o.location_id = ol.location_id

    left join received_locations rl
        on r.received_at_location_id = rl.location_id

)

select *
from final
