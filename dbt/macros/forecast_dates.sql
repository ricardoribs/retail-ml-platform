{% macro forecast_cutoff_date() -%}
    last_day(
        add_months(
            {{ var("forecast_config")["reference_date_sql"] }},
            -1
        )
    )
{%- endmacro %}
