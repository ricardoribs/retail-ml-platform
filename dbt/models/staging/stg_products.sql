with source_data as (
    
    select * from {{ source('bronze', 'products') }}
),
cleaned_data as (

    select
        cast(id as bigint)                as product_id,
        cast(name as string)              as product_name,
        cast(description as string)       as description,
        cast(brand_id as bigint)          as brand_id,
        cast(category_id as bigint)       as category_id,
        cast(ncm_code as bigint)          as ncm_code,
        cast(unit_of_measure as string)   as unit_of_measure,
        cast(is_active as boolean)        as is_active,
        cast(created_at as timestamp)     as created_at,
        cast(updated_at as timestamp)     as updated_at
    from source_data
),

deduped_data as (
    
    select
        *,
        row_number() over (
            partition by product_id
            order by updated_at desc
        ) as rn
    from cleaned_data
),
final as (

    select
        product_id,
        product_name,
        description,
        brand_id,
        category_id,
        ncm_code,
        unit_of_measure,
        is_active,
        created_at,
        updated_at
    from deduped_data
    where rn = 1
)

select * from final
