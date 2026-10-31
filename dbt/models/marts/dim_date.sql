with date_spine as (

    {{ dbt_utils.date_spine(
        datepart="day",
        start_date="cast('2020-01-01' as date)",
        end_date="cast('2028-01-01' as date)"
    ) }}

),

date_attributes as (

    select
        cast(date_format(date_day, 'yyyyMMdd') as int) as date_key,
        cast(date_day as date)                          as full_date,
        year(date_day)                                  as year_number,
        quarter(date_day)                               as quarter_number,
        month(date_day)                                 as month_number,
        date_format(date_day, 'MMMM')                   as month_name,
        day(date_day)                                   as day_number,
        weekofyear(date_day)                            as week_number,
        date_format(date_day, 'EEEE')                   as day_of_week,
        {{ flag_covid_period('date_day') }}             as is_covid_period
    from date_spine

),

final as (

    select
        date_key,
        full_date,
        year_number,
        quarter_number,
        month_number,
        month_name,
        day_number,
        week_number,
        day_of_week,
        is_covid_period
    from date_attributes

)

select * from final
