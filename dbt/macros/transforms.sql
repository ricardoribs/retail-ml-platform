{% macro get_return_reason_class(return_reason_column) %}
    case
        when {{ return_reason_column }} is null then 'unclassified'

        -- Operational failure: damaged product, defect, or shipping issue.
        when lower(trim({{ return_reason_column }})) like '%avaria%'
          or lower(trim({{ return_reason_column }})) like '%defeito%'
          or lower(trim({{ return_reason_column }})) like '%transporte%'
        then 'operational_failure'

        -- Information mismatch: incorrect size/color or item does not match its description.
        when lower(trim({{ return_reason_column }})) like '%tamanho%'
          or lower(trim({{ return_reason_column }})) like '%descri%'
        then 'information_mismatch'

        -- Consumer behavior: changed mind or duplicate purchase.
        when lower(trim({{ return_reason_column }})) like '%desist%'
          or lower(trim({{ return_reason_column }})) like '%duplicad%'
        then 'consumer_behavior'

        else 'unclassified'
    end
{% endmacro %}

{% macro flag_covid_period(date_column) %}
    -- Marks dates within the pandemic period (2020-2021) for seasonality and COVID impact analyses.
    case
        when {{ date_column }} between '2020-01-01' and '2021-12-31'
        then true
        else false
    end
{% endmacro %}
