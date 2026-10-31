with source_data as (

    select * from {{ source('bronze', 'stock_movements') }}

),

cleaned_data as (

    select
        cast(id as bigint)                as stock_movement_id,
        cast(product_variant_id as bigint) as product_variant_id,
        cast(location_id as bigint)        as location_id,
        cast(movement_type as string)      as movement_type,
        cast(quantity as decimal(18,3))    as quantity,
        cast(reference_table as string)    as reference_table,
        cast(reference_id as bigint)       as reference_id,
        cast(employee_id as bigint)        as employee_id,
        cast(notes as string)              as notes,
        cast(occurred_at as timestamp)     as occurred_at,
        cast(created_at as timestamp)      as created_at
    from source_data

),

deduped_data as (

    select
        *,
        row_number() over (
            partition by stock_movement_id
            order by stock_movement_id
        ) as rn
    from cleaned_data

),

final as (

    select
        stock_movement_id,
        product_variant_id,
        location_id,
        movement_type,
        quantity,
        reference_table,
        reference_id,
        employee_id,
        notes,
        occurred_at,
        created_at
    from deduped_data
    where rn = 1

)

select * from final
