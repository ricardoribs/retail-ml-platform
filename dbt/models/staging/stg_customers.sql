with source_data as (

    select *
    from {{ source('bronze', 'customers') }}

),

cleaned_data as (

    select
        cast(id as bigint) as customer_id,
        person_type,
        legal_name as customer_name,
        trade_name,
        tax_id,
        state_registration,
        lower(trim(email)) as email,
        phone,
        cast(is_active as boolean) as is_active_flag,
        cast(created_at as timestamp) as created_at,
        cast(updated_at as timestamp) as updated_at

    from source_data

),

deduped_data as (

    select *
    from cleaned_data
    qualify row_number() over (
        partition by customer_id
        order by updated_at desc
    ) = 1

),

final as (

    select *
    from deduped_data

)

select *
from final
