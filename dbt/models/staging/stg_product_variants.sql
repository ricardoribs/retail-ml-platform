with source_data as (

    select * from {{ source('bronze', 'product_variants') }}

),

cleaned_data as (

    select
        cast(id                as bigint)         as product_variant_id,
        cast(product_id        as bigint)         as product_id,
        cast(sku               as string)         as sku,
        cast(barcode_ean       as string)         as barcode_ean,
        cast(sale_price        as decimal(18,2))  as sale_price,
        cast(cost_price        as decimal(18,2))  as cost_price,
        cast(weight_kg         as decimal(10,3))  as weight_kg,
        cast(icms_rate         as decimal(5,2))   as icms_rate,
        cast(ipi_rate          as decimal(5,2))   as ipi_rate,
        cast(is_active         as boolean)        as is_active,
        cast(created_at        as timestamp)      as created_at,
        cast(updated_at        as timestamp)      as updated_at

    from source_data

),

final as (

    select * from cleaned_data

)

select * from final
