with orders as (

    select *
    from {{ ref('stg_orders') }}

),

final as (

    select
        order_id,
        coalesce(salesperson_id, -1) as salesperson_id,
        case
            when salesperson_id is null then 'E-commerce (Autoatendimento)'
            else 'Vendedor Regular'
        end as salesperson_type
    from orders

)

select *
from final
