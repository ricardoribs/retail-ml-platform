with locations as (

    select *
    from {{ ref('stg_locations') }}

),

final as (

    select
        {{ dbt_utils.generate_surrogate_key(['location_id']) }} as location_key,
        location_id,
        location_name,
        location_type,
        city,
        state,
        country,
        is_active_flag
    from locations

)

select *
from final
