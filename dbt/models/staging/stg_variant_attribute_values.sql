with source_data as (

    select * from {{ source('bronze', 'variant_attribute_values') }}

),

cleaned_data as (

    select
        cast(product_variant_id as bigint) as product_variant_id,
        cast(attribute_id       as bigint) as attribute_id,
        cast(value              as string) as attribute_value

    from source_data

),

final as (

    select * from cleaned_data

)

select * from final
