with employees as (

    select * from {{ ref('stg_employees') }}

),

ecommerce_record as (

    -- Synthetic record representing sales without a human salesperson (e-commerce).
    select
        cast(-1 as string) as employee_key,
        cast(-1 as bigint) as employee_id,
        'E-commerce (Autoatendimento)' as employee_name,
        cast(null as string) as role,
        cast(null as bigint) as location_id,
        cast(null as date) as hire_date,
        cast(null as boolean) as is_active_flag

),

regular_employees as (

    select
        {{ dbt_utils.generate_surrogate_key(['employee_id']) }} as employee_key,
        employee_id,
        employee_name,
        role,
        location_id,
        hire_date,
        is_active_flag
    from employees

),

final as (

    select * from ecommerce_record
    union all
    select * from regular_employees

)

select * from final
