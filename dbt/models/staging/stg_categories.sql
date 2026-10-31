with source_data as (
 
    select * from {{ source('bronze', 'categories') }}
),

cleaned_data as (
    
    select
        cast(id as bigint)                    as category_id,
        cast(name as string)                  as category_name,
        cast(slug as string)                  as slug,
        cast(parent_category_id as bigint)    as parent_category_id,
        cast(is_active as boolean)            as is_active,
        cast(created_at as timestamp)         as created_at,
        cast(updated_at as timestamp)         as updated_at
    from source_data
),

deduped_data as (

    select
        *,
        row_number() over (
            partition by category_id
            order by updated_at desc
        ) as rn
    from cleaned_data
),

final as (

    select
        category_id,
        category_name,
        slug,
        parent_category_id,
        is_active,
        created_at,
        updated_at
    from deduped_data
    where rn = 1
)

select * from final
