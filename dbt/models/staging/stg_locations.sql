with source_data as (

    select *
    from {{ source('bronze', 'locations') }}

),

cleaned_data as (

    select
        cast(id as bigint) as location_id,
        name as location_name,
        location_type,
        postal_code,
        street,
        number,
        complement,
        district,
        city,
        state,
        country,
        cast(is_active as boolean) as is_active_flag,
        cast(created_at as timestamp) as created_at,
        cast(updated_at as timestamp) as updated_at

    from source_data

),

deduped_data as (

    select *
    from cleaned_data
    qualify row_number() over (
        partition by location_id
        order by updated_at desc
    ) = 1

),

final as (

    select *
    from deduped_data

)

select *
from final
