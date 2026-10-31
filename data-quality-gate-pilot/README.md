# Data Quality Gate dbt pilot

This is an isolated Databricks bundle for the Data Quality Gate V2 exit-gate pilot.
It is intentionally separate from the repository-level `databricks.yml`, which
defines the historical ingestion, forecasting, and Marina workflow.

## Safety boundaries

- The bundle declares one manual Job: `data_quality_gate_dbt_pilot`.
- It has no schedule or trigger and runs at most once at a time.
- The runner permits only the `retail_analytics` catalog and schemas beginning with
  `data_quality_gate_pilot_`.
- It executes only `dbt deps` and `dbt test --select path:tests/pilot`.
- The dbt profile is created in a temporary directory and deleted after the run.
- Local `dbt/profiles.yml`, generated dbt files, and secrets are excluded from
  bundle synchronization.

## Validation before deployment

Run these commands from this directory after reviewing the branch:

```powershell
databricks bundle validate -t dev --profile <dev-profile>
```

Use the profile name configured in your local Databricks CLI setup. The profile
is intentionally not stored in `databricks.yml`, so the bundle remains portable
and contains no local authentication configuration.

Do not deploy the repository-level bundle for this pilot. A later, explicit
approval is required before using `databricks bundle deploy -t dev` here.

## Controlled failure sequence

The baseline `dbt/tests/pilot/data_quality_gate_smoke.sql` returns zero rows and
passes. After the baseline run is recorded, a dedicated failure branch changes
that single fixture to return one row. The nonzero dbt test result is the
incident used to demonstrate policy blocking. The approved corrective patch
then restores the zero-row query for the validated draft-PR and idempotency
checks.
