-- This fixture returns zero rows unless the isolated pilot explicitly requests
-- a controlled failure. It is deterministic and independent of business data.
{% if var('data_quality_gate_inject_failure', false) %}
select 'data quality gate controlled failure' as reason
{% else %}
select 'data quality gate pilot baseline unexpectedly returned a row' as reason
where 1 = 1
{% endif %}
