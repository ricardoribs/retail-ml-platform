with source_data as (

    select * from {{ source('bronze', 'fiscal_invoices') }}

),

cleaned_data as (

    select
        cast(id as bigint)                as fiscal_invoice_id,
        cast(order_id as bigint)          as order_id,
        cast(nfe_number as string)        as invoice_number,
        cast(nfe_access_key as string)    as nfe_access_key,
        cast(series as string)            as series,
        cast(issued_at as timestamp)      as issued_at,
        cast(status as string)            as invoice_status,
        cast(total_amount as decimal(18,2)) as total_amount,
        cast(xml_storage_uri as string)   as xml_storage_uri
    from source_data

),

deduped_data as (

    select
        *,
        row_number() over (
            partition by fiscal_invoice_id
            order by issued_at desc
        ) as rn
    from cleaned_data

),

final as (

    select
        fiscal_invoice_id,
        order_id,
        invoice_number,
        nfe_access_key,
        series,
        issued_at,
        invoice_status,
        total_amount,
        xml_storage_uri
    from deduped_data
    where rn = 1

)

select * from final
