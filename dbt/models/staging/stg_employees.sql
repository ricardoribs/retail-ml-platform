with source_data as (

    select *
    from {{ source('bronze', 'employees') }}

),

cleaned_data as (

    select
        cast(id as bigint) as employee_id,
        full_name as employee_name,
        cpf,
        lower(trim(email)) as email,
        role,
        cast(primary_location_id as bigint) as location_id,
        cast(hire_date as date) as hire_date,
        cast(termination_date as date) as termination_date,
        cast(is_active as boolean) as is_active_flag,
        cast(created_at as timestamp) as created_at,
        cast(updated_at as timestamp) as updated_at

    from source_data

),

deduped_data as (

    select *
    from cleaned_data
    qualify row_number() over (
        partition by employee_id
        order by updated_at desc
    ) = 1

),

final as (

    select *
    from deduped_data

)

select *
from final
