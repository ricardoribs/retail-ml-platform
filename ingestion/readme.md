# Bronze data ingestion

This module demonstrates ingestion from SQL Server into the Databricks Bronze layer. It supports full and incremental loads, idempotent operations, watermark tracking, and automated execution through Databricks Workflows. Adjust source and catalog settings for each runtime environment.

## Contents

1. [Configure ingestion](#1-configure-ingestion)
2. [Run manually in Databricks](#2-run-manually-in-databricks)
3. [Add or change a table](#3-add-or-change-a-table)
4. [Deploy](#4-deploy)
5. [Structure](#5-structure)
6. [Workflow](#6-workflow)
7. [Monitoring](#7-monitoring)
8. [Notes](#8-notes)

## 1. Configure ingestion

Create a Databricks Secret Scope named retail-analytics-source with these SQL Server credentials:

~~~text
db-host
db-port
db-name
db-user
db-password
~~~

Review config/tables_config.json. Its general settings include the source schema, secret scope, Bronze catalog and schema, and control table.

The workspace must have network access to SQL Server and permission to create and update tables in the retail_analytics catalog.

## 2. Run manually in Databricks

To validate or run ingestion manually:

1. Open notebooks/00_setup_bronze.py.
2. Confirm that the config_path widget points to ../config/tables_config.json.
3. Run the notebook to check configuration, Secret Scope, control table, and JDBC connectivity.
4. Open notebooks/01_bronze_ingestion.py.
5. Set table_name to all or provide one table name.
6. Run the notebook and review its final summary.

The main notebook fails if any table cannot be processed. It continues processing the remaining tables so the summary can report all failures.

## 3. Add or change a table

Tables are declared in config/tables_config.json. Choose one of these load strategies:

| Strategy | Use |
|---|---|
| full | Small tables that can be fully overwritten. |
| merge | Tables that receive inserts and updates. |
| merge_insert_only | Append-only tables whose existing rows must not be updated. |

When adding a table:

1. Confirm its source name and schema.
2. Choose a stable primary key.
3. Set a watermark column for incremental loading when applicable.
4. Use partitioned reads only for tables with significant volume.
5. Run 00_setup_bronze.py.
6. Test the new table with the table_name widget.
7. Compare row counts and keys between the source and Bronze.

## 4. Deploy

This module is part of the root Databricks Asset Bundle. From the repository root:

~~~bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run bronze_ingestion -t dev
~~~

The bundle resource key remains bronze_ingestion, while Databricks displays the job as retail_analytics_data_pipeline. The workflow includes Bronze, Silver, Gold, forecasting, logistics rankings, and Marina document indexing.

For production, replace dev with prod only after validating the run in development.

## 5. Structure

~~~text
ingestion/
├── bundle/                         # Bronze-to-dbt-to-forecast workflow
├── config/                         # Source tables and load strategies
├── notebooks/
│   ├── 00_setup_bronze.py          # Environment validation
│   └── 01_bronze_ingestion.py      # Ingestion run
├── src/
│   ├── connection.py               # JDBC connection
│   ├── control.py                  # Control table and watermarks
│   └── ingestion.py                # Table reads and writes
└── readme.md
~~~

Module responsibilities:

- connection.py reads secrets, configures JDBC, and checks SQL Server access.
- control.py creates and updates ingestion_control and retrieves the latest watermark.
- ingestion.py selects data, adds metadata, and applies full, merge, or merge_insert_only.

## 6. Workflow

The job in bundle/bronze_ingestion_job.yml runs daily at 8:00 AM São Paulo time. The diagram summarizes the workflow; exact dependencies and parallel branches are declared in the bundle.

~~~mermaid
flowchart LR
    A[setup_bronze] --> B[bronze_ingestion]
    B --> C[dbt_transformation]
    C --> D[SKU-location forecasting]
    D --> E[Location-level planning]
    E --> F[Rankings, screening, and Marina]
~~~

- setup_bronze validates configuration, the control table, and connectivity.
- bronze_ingestion processes configured tables and updates watermarks.
- dbt_transformation runs Silver and Gold transformations after Bronze succeeds.
- Later tasks evaluate and publish SKU-location forecasts, create location-level aggregates, calculate rankings, and synchronize Marina's document index.

The job allows one concurrent run, has up to two additional retries per task, and sends email on failure.

## 7. Monitoring

Check the control table for the latest run of each table:

~~~sql
select
    table_name,
    last_watermark_value,
    last_run_at,
    last_status,
    rows_processed,
    error_message
from retail_analytics.bronze.ingestion_control
order by last_run_at desc;
~~~

Find stale loads:

~~~sql
select
    table_name,
    last_run_at,
    last_status
from retail_analytics.bronze.ingestion_control
where last_run_at < current_timestamp() - interval 2 days;
~~~

Task logs, retries, and dependencies are available in Databricks Workflows.

## 8. Notes

- A watermark advances only after a successful write.
- A failed run preserves the last valid watermark for reprocessing.
- Set table_name=<name> to test or reprocess one table.
- Prefer indexed, non-null watermark columns.
- Validate schema changes before running a production merge.
- Never store credentials or tokens in notebooks, JSON files, or version-controlled files.
- Later transformations are documented in the [dbt guide](../dbt/readme.md) and the [forecasting guide](../ml/demand_forecasting/README.md).