with source_data as (

    select * from {{ source('bronze', 'attributes') }}
),

cleaned_data as (
    select
        cast(id as bigint)              as attribute_id,
        cast(name as string)            as attribute_name,
        cast(data_type as string)       as data_type
    from source_data
),

deduped_data as (

    select
        *,
        row_number() over (
            partition by attribute_id
            order by attribute_id
        ) as rn
    from cleaned_data
),

final as (

    select
        attribute_id,
        attribute_name,
        data_type
    from deduped_data
    where rn = 1
)

select * from final
