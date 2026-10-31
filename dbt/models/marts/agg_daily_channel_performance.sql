{{ config(materialized='table') }}

-- Selects orders and the key attributes used in the aggregation.
with orders as (

    select
        order_id,
        customer_id,
        channel,
        order_status,
        subtotal,
        discount_amount,
        total_amount,
        placed_at
    from {{ ref('stg_orders') }}

),

-- Selects each order's items to calculate quantities and sales amounts.
order_items as (

    select
        order_item_id,
        order_id,
        quantity_qty,
        line_total
    from {{ ref('stg_order_items') }}

),

-- Consolidates item metrics at the order level.
order_item_totals as (

    select
        order_id,
        count(distinct order_item_id) as item_line_count,
        sum(quantity_qty) as quantity_sold_qty,
        sum(line_total) as item_gross_amount_brl
    from order_items
    group by order_id

),

-- Retrieves the date dimension key.
dates as (

    select
        date_key,
        full_date
    from {{ ref('dim_date') }}

),

-- Enriches orders with item metrics and the date dimension key.
orders_enriched as (

    select
        o.order_id,
        o.customer_id,
        o.channel,
        o.order_status,
        d.date_key,

        coalesce(oit.item_line_count, 0) as item_line_count,
        coalesce(oit.quantity_sold_qty, 0) as quantity_sold_qty,
        coalesce(oit.item_gross_amount_brl, 0) as item_gross_amount_brl,

        coalesce(o.subtotal, 0) as subtotal_amount_brl,
        coalesce(o.discount_amount, 0) as discount_amount_brl,
        coalesce(o.total_amount, 0) as net_revenue_amount_brl

    from orders o

    left join order_item_totals oit
        on o.order_id = oit.order_id

    left join dates d
        on cast(o.placed_at as date) = d.full_date

),

-- Aggregates sales metrics by day and channel.
daily_channel_totals as (

    select
        date_key,
        channel,

        count(distinct order_id) as total_order_count,

        count(
            distinct case
                when order_status = 'paid' then order_id
            end
        ) as paid_order_count,

        count(
            distinct case
                when order_status = 'cancelled' then order_id
            end
        ) as cancelled_order_count,

        count(
            distinct case
                when order_status = 'confirmed' then order_id
            end
        ) as confirmed_order_count,

        count(
            distinct case
                when order_status = 'draft' then order_id
            end
        ) as draft_order_count,

        count(
            distinct case
                when order_status = 'paid' then customer_id
            end
        ) as paid_customer_count,

        sum(
            case
                when order_status = 'paid' then item_line_count
                else 0
            end
        ) as item_line_count,

        sum(
            case
                when order_status = 'paid' then quantity_sold_qty
                else 0
            end
        ) as quantity_sold_qty,

        sum(
            case
                when order_status = 'paid' then subtotal_amount_brl
                else 0
            end
        ) as gross_revenue_amount_brl,

        sum(
            case
                when order_status = 'paid' then discount_amount_brl
                else 0
            end
        ) as discount_amount_brl,

        sum(
            case
                when order_status = 'paid' then net_revenue_amount_brl
                else 0
            end
        ) as net_revenue_amount_brl

    from orders_enriched

    group by
        date_key,
        channel

),

-- Calculate the main commercial performance KPIs.
performance_indicators as (

    select
        *,

        case
            when total_order_count = 0 then 0
            else cast(paid_order_count as decimal(18, 4))
                / total_order_count
        end as paid_conversion_rate,

        case
            when total_order_count = 0 then 0
            else cast(cancelled_order_count as decimal(18, 4))
                / total_order_count
        end as cancellation_rate,

        case
            when paid_order_count = 0 then 0
            else net_revenue_amount_brl
                / paid_order_count
        end as average_order_value_brl,

        case
            when paid_order_count = 0 then 0
            else discount_amount_brl
                / paid_order_count
        end as average_discount_per_order_brl,

        case
            when paid_customer_count = 0 then 0
            else net_revenue_amount_brl
                / paid_customer_count
        end as revenue_per_customer_brl,

        case
            when paid_order_count = 0 then 0
            else cast(quantity_sold_qty as decimal(18, 4))
                / paid_order_count
        end as average_items_per_order

    from daily_channel_totals

),

-- Calculates each channel's share of net revenue for the day.
channel_share as (

    select
        *,

        case
            when sum(net_revenue_amount_brl) over (
                partition by date_key
            ) = 0 then 0

            else net_revenue_amount_brl
                / sum(net_revenue_amount_brl) over (
                    partition by date_key
                )
        end as channel_revenue_share

    from performance_indicators

),

-- Builds the final structure of the aggregated fact table.
final as (

    select
        {{ dbt_utils.generate_surrogate_key([
            'date_key',
            'channel'
        ]) }} as daily_channel_performance_key,

        date_key,
        channel,

        total_order_count,
        paid_order_count,
        cancelled_order_count,
        confirmed_order_count,
        draft_order_count,
        paid_customer_count,

        item_line_count,
        quantity_sold_qty,

        gross_revenue_amount_brl,
        discount_amount_brl,
        net_revenue_amount_brl,

        paid_conversion_rate,
        cancellation_rate,
        average_order_value_brl,
        average_discount_per_order_brl,
        revenue_per_customer_brl,
        average_items_per_order,
        channel_revenue_share

    from channel_share

)

select *
from final
