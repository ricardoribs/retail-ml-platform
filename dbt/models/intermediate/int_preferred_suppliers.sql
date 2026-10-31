with suppliers as (
    select
        product_variant_id,
        supplier_id,
        lead_time_days,
        last_quoted_cost,
        is_preferred,
        updated_at,
        
        row_number() over (
            partition by product_variant_id
            order by 
                is_preferred desc,
                updated_at desc,
                lead_time_days asc,
                supplier_id asc
        ) as rn
        
    from {{ ref('stg_product_suppliers') }}
),

preferred as (
    select
        product_variant_id,
        supplier_id,
        lead_time_days,
        last_quoted_cost,
        is_preferred,
        updated_at
    from suppliers
    where rn = 1
)

select * from preferred
