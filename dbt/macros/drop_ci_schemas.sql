{% macro drop_ci_schemas(pr_id) %}
    {#
        Removes temporary schemas created by the pull request pipeline.
        Execution is restricted to the ci_env target to prevent accidental
        removal in development or production environments.
    #}
    {% if target.name != "ci_env" %}
        {{ exceptions.raise_compiler_error(
            "drop_ci_schemas can only run with the ci_env target."
        ) }}
    {% endif %}

    {% set normalized_pr_id = pr_id | int %}
    {% set schema_prefix = "ci_pr_" ~ normalized_pr_id %}
    {% set ci_schemas = [
        schema_prefix,
        schema_prefix ~ "_staging",
        schema_prefix ~ "_intermediate",
        schema_prefix ~ "_marts",
        schema_prefix ~ "_test_failures"
    ] %}

    {% for schema_name in ci_schemas %}
        {% set drop_schema_sql %}
            drop schema if exists `{{ target.catalog }}`.`{{ schema_name }}` cascade
        {% endset %}

        {% do run_query(drop_schema_sql) %}
    {% endfor %}
{% endmacro %}
