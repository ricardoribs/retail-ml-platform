with source_data as (

    select * from {{ source('bronze', 'suppliers') }}

),

cleaned_data as (

    select
        cast(id as bigint)             as supplier_id,
        cast(legal_name as string)     as legal_name,
        cast(trade_name as string)     as trade_name,
        cast(country as string)        as country,
        cast(tax_id as string)         as tax_id,
        cast(tax_id_type as string)    as tax_id_type,
        cast(email as string)          as email,
        cast(phone as string)          as phone,
        cast(contact_name as string)   as contact_name,
        cast(is_active as boolean)     as is_active,
        cast(created_at as timestamp)  as created_at,
        cast(updated_at as timestamp)  as updated_at
    from source_data

),

deduped_data as (

    -- Remove duplicates, keeping one row per supplier_id.
    select
        *,
        row_number() over (
            partition by supplier_id
            order by updated_at desc
        ) as rn
    from cleaned_data

),

final as (

    select
        supplier_id,
        legal_name,
        trade_name,
        country,
        tax_id,
        tax_id_type,
        email,
        phone,
        contact_name,
        is_active,
        created_at,
        updated_at
    from deduped_data
    where rn = 1

)

select * from final
