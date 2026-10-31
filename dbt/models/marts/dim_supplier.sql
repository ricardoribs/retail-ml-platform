with suppliers as (

    select * from {{ ref('stg_suppliers') }}

),

supplier_lead_times as (

    select
        ps.supplier_id,
        avg(datediff(gr.received_at, po.created_at)) as average_lead_time_days
    from {{ ref('stg_purchase_orders') }} po
    join {{ ref('stg_goods_receipts') }} gr on po.purchase_order_id = gr.purchase_order_id
    join {{ ref('stg_product_suppliers') }} ps on po.supplier_id = ps.supplier_id
    group by ps.supplier_id

),

final as (

    select
        {{ dbt_utils.generate_surrogate_key(['s.supplier_id']) }} as supplier_key,
        s.supplier_id,
        s.legal_name,
        s.trade_name,
        s.country,
        s.tax_id,
        s.tax_id_type,
        s.email,
        s.phone,
        s.contact_name,
        s.is_active,
        slt.average_lead_time_days
    from suppliers s
    left join supplier_lead_times slt
        on s.supplier_id = slt.supplier_id

)

select * from final
