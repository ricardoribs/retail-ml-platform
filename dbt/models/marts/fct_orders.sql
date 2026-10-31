with orders as (

    select * from {{ ref('stg_orders') }}

),

payment_reconciliation as (

    select * from {{ ref('int_payment_reconciliation') }}

),

order_items_aggregated as (

    select
        order_id,
        count(distinct order_item_id) as total_items,
        sum(quantity_qty) as total_units,
        sum(line_total) as total_line_amount_brl,
        sum(line_total * (icms_rate + ipi_rate) / 100) as total_tax_amount_brl
    from {{ ref('stg_order_items') }}
    group by order_id

),

orders_enriched as (

    select
        o.order_id,
        o.order_number,
        {{ dbt_utils.generate_surrogate_key(['o.customer_id']) }} as customer_key,
        {{ dbt_utils.generate_surrogate_key(['o.location_id']) }} as location_key,
        case
            when o.salesperson_id is null then cast(-1 as string)
            else {{ dbt_utils.generate_surrogate_key(['o.salesperson_id']) }}
        end as employee_key,
        cast(o.placed_at as date) as order_date,
        o.placed_at as order_datetime,
        o.channel,
        o.order_status,
        o.subtotal as order_subtotal_brl,
        o.discount_amount as order_discount_amount_brl,
        o.total_amount as order_total_amount_brl,
        coalesce(oi.total_items, 0) as total_items,
        coalesce(oi.total_units, 0) as total_units,
        coalesce(oi.total_line_amount_brl, 0) as total_line_amount_brl,
        coalesce(oi.total_tax_amount_brl, 0) as total_tax_amount_brl,
        coalesce(pr.total_paid_amount_brl, 0) as total_paid_amount_brl,
        coalesce(pr.total_refunded_amount_brl, 0) as total_refunded_amount_brl,
        coalesce(pr.total_pending_amount_brl, 0) as total_pending_amount_brl,
        coalesce(pr.total_refused_amount_brl, 0) as total_refused_amount_brl,
        coalesce(pr.raw_net_revenue_amount_brl, 0) as raw_net_revenue_amount_brl,
        coalesce(pr.recognized_net_revenue_amount_brl, 0) as recognized_net_revenue_amount_brl,
        coalesce(pr.unmatched_refund_amount_brl, 0) as unmatched_refund_amount_brl,
        coalesce(pr.recognized_refund_amount_brl, 0) as recognized_refund_amount_brl,
        coalesce(pr.pending_potential_value_brl, 0) as pending_potential_value_brl,
        coalesce(pr.approved_payments_count, 0) as approved_payments_count,
        coalesce(pr.refunded_payments_count, 0) as refunded_payments_count,
        coalesce(pr.pending_payments_count, 0) as pending_payments_count,
        coalesce(pr.refused_payments_count, 0) as refused_payments_count,
        coalesce(pr.total_payments_count, 0) as total_payments_count,
        coalesce(pr.has_approved_payment_flag, false) as has_approved_payment_flag,
        coalesce(pr.has_refund_flag, false) as has_refund_flag,
        coalesce(pr.has_pending_payment_flag, false) as has_pending_payment_flag,
        coalesce(pr.has_refused_payment_flag, false) as has_refused_payment_flag,
        coalesce(pr.has_refund_without_approved_payment_flag, false) as has_refund_without_approved_payment_flag,
        coalesce(pr.has_refund_exceeding_paid_flag, false) as has_refund_exceeding_paid_flag,
        (pr.order_id is null) as has_no_payment_record_flag,
        coalesce(pr.payment_reconciliation_status, 'no_financial_movement') as payment_reconciliation_status,
        (o.order_status = 'pending') as is_pending_order_flag,
        (o.order_status = 'cancelled') as is_cancelled_order_flag,
        (o.order_status in ('completed', 'shipped', 'delivered')) as is_completed_order_flag,
        datediff(day, cast(o.placed_at as date), current_date()) as days_since_placed,
        case
            when o.order_status in ('completed', 'shipped', 'delivered') 
                and pr.order_id is null 
            then true
            else false
        end as payment_anomaly_flag,
        case
            when coalesce(pr.has_approved_payment_flag, false) = true
                and o.order_status not in ('pending', 'cancelled')
                and coalesce(pr.has_refund_without_approved_payment_flag, false) = false
            then true
            else false
        end as is_eligible_for_kpis_flag
    from orders o
    left join order_items_aggregated oi
        on o.order_id = oi.order_id
    left join payment_reconciliation pr
        on o.order_id = pr.order_id

)

select * from orders_enriched
