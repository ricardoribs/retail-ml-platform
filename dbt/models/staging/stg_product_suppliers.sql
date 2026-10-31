with source_data as (

    select * from {{ source('bronze', 'product_suppliers') }}

),

cleaned_data as (

    select
        cast(product_variant_id as bigint)      as product_variant_id,
        cast(supplier_id as bigint)              as supplier_id,
        cast(supplier_sku as string)             as supplier_sku,
        cast(last_quoted_cost as decimal(18,2))  as last_quoted_cost,
        cast(lead_time_days as int)              as lead_time_days,
        cast(is_preferred as boolean)            as is_preferred,
        cast(created_at as timestamp)            as created_at,
        cast(updated_at as timestamp)            as updated_at
    from source_data

),

deduped_data as (

    -- Remove duplicates, keeping one row per product_variant_id + supplier_id.
    select
        *,
        row_number() over (
            partition by product_variant_id, supplier_id
            order by updated_at desc
        ) as rn
    from cleaned_data

),

final as (

    select
        product_variant_id,
        supplier_id,
        supplier_sku,
        last_quoted_cost,
        lead_time_days,
        is_preferred,
        created_at,
        updated_at
    from deduped_data
    where rn = 1

)

select * from final
