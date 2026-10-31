with source_data as (

    select *
    from {{ source('bronze', 'addresses') }}

),

cleaned_data as (

    select
        cast(id as bigint) as address_id,
        cast(customer_id as bigint) as customer_id,
        address_type,
        postal_code,
        street,
        number,
        complement,
        district,
        city,
        state,
        country,
        cast(is_primary as boolean) as is_primary_flag

    from source_data

),

deduped_data as (

    select *
    from cleaned_data
    qualify row_number() over (
        partition by address_id
        order by customer_id
    ) = 1

),

final as (

    select *
    from deduped_data

)

select *
from final
