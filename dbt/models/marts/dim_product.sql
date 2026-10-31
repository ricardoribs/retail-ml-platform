with variants as (

    select * from {{ ref('stg_product_variants') }}

),

products as (

    select * from {{ ref('stg_products') }}

),

brands as (

    select * from {{ ref('stg_brands') }}

),

categories as (

    select * from {{ ref('stg_categories') }}

),

-- Flattens the hierarchy by joining each category to its parent name through parent_category_id.
categories_hierarchy as (

    select
        c.category_id,
        c.category_name,
        p.category_name as parent_category_name
    from categories c
    left join categories p
        on c.parent_category_id = p.category_id

),

joined as (

    select
        v.product_variant_id,
        v.product_id,
        v.sku,
        v.barcode_ean,
        v.sale_price     as sale_price_amount,
        v.cost_price     as cost_price_amount,
        v.weight_kg,
        v.icms_rate,
        v.ipi_rate,
        p.product_name,
        p.ncm_code,
        p.unit_of_measure,
        b.brand_id,
        b.brand_name,
        ch.category_id,
        ch.category_name,
        ch.parent_category_name,
        v.is_active as is_active_flag,
        v.created_at,
        v.updated_at
    from variants v
    left join products p
        on v.product_id = p.product_id
    left join brands b
        on p.brand_id = b.brand_id
    left join categories_hierarchy ch
        on p.category_id = ch.category_id

),

final as (

    select
        {{ dbt_utils.generate_surrogate_key(['product_variant_id']) }} as product_key,
        product_variant_id,
        product_id,
        sku,
        barcode_ean,
        product_name,
        brand_id,
        brand_name,
        category_id,
        category_name,
        parent_category_name,
        ncm_code,
        unit_of_measure,
        sale_price_amount,
        cost_price_amount,
        weight_kg,
        icms_rate,
        ipi_rate,
        is_active_flag,
        created_at,
        updated_at
    from joined

)

select * from final
