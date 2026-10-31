with source_data as (
   
    select * from {{ source('bronze', 'brands') }}
),

cleaned_data as (

    select
        cast(id as bigint)               as brand_id,
        cast(name as string)             as brand_name,
        cast(country as string)          as country,
        cast(is_active as boolean)       as is_active,
        cast(created_at as timestamp)    as created_at,
        cast(updated_at as timestamp)    as updated_at
    from source_data
),
deduped_data as (
    
    select
        *,
        row_number() over (
            partition by brand_id
            order by updated_at desc
        ) as rn
    from cleaned_data
),

final as (

    select
        brand_id,
        brand_name,
        country,
        is_active,
        created_at,
        updated_at
    from deduped_data
    where rn = 1
)

select * from final
