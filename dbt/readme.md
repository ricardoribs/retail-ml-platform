# dbt analytics modeling

This module transforms Bronze data into Silver and Gold models in Databricks. It uses dbt to standardize data, apply analytical rules, build dimensional tables, and run quality tests that support analytics and forecasting experiments.

## Contents

1. [Set up dbt](#1-set-up-dbt)
2. [Run dbt](#2-run-dbt)
3. [Create or change a model](#3-create-or-change-a-model)
4. [Testing](#4-testing)
5. [Deployment](#5-deployment)
6. [Structure](#6-structure)
7. [Workflow](#7-workflow)
8. [Notes](#8-notes)

## 1. Set up dbt

### Prerequisites

- Python 3.12 or a compatible version.
- Access to a Databricks SQL Warehouse.
- Read access to retail_analytics.bronze.
- Permission to create views and tables in the target schemas.

Create and activate a virtual environment from the repository root:

~~~powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install dbt-databricks
~~~

Install dbt packages:

~~~powershell
cd dbt
dbt deps
~~~

Use profiles.yml.example to create a local profiles.yml. Set the host and HTTP path, and provide the token only through an environment variable:

~~~powershell
$env:DATABRICKS_TOKEN = "<your-token>"
~~~

Check the connection:

~~~powershell
dbt debug --target dev
~~~

## 2. Run dbt

From the dbt directory:

~~~bash
dbt deps
dbt compile --target dev
dbt run --target dev
dbt test --target dev
~~~

Build models and run tests in one command:

~~~bash
dbt build --target dev
~~~

Selective runs:

~~~bash
# Staging models only
dbt run --select path:models/staging --target dev

# One model and all of its upstream dependencies
dbt build --select +fct_sales --target dev

# Singular tests only
dbt test --select test_type:singular --target dev
~~~

## 3. Create or change a model

Choose a layer based on the model's responsibility:

| Layer | Purpose | Materialization |
|---|---|---|
| staging | Clean, type, rename, and deduplicate Bronze sources. | View |
| intermediate | Reusable business rules. | View |
| marts | Dimensions, facts, and analytical aggregates. | Table |

When adding a model:

1. Use source() to read a Bronze table.
2. Use ref() to reference another dbt model.
3. Define and preserve the model grain.
4. Follow the stg_, int_, dim_, fct_, or agg_ prefixes.
5. Document the model and its columns in the layer's schema.yml.
6. Add tests that match the data rules.
7. Run dbt compile and dbt build --select +<model>.

Example staging model:

~~~sql
with source_data as (

    select *
    from {{ source('bronze', 'orders') }}

),

final as (

    select
        cast(id as bigint) as order_id,
        cast(placed_at as timestamp) as placed_at
    from source_data

)

select * from final
~~~

## 4. Testing

The schema.yml files define generic tests:

- not_null: the field must not be null.
- unique: the key must not be duplicated.
- accepted_values: the value must belong to the declared domain.
- relationships: the key must exist in the related model.

Singular tests are in tests/business_logic/. A test query should return only invalid records; zero rows means it passed.

Run a specific test:

~~~bash
dbt test --select test_payment_reconciliation_accuracy --target dev
~~~

Generate documentation and view lineage:

~~~bash
dbt docs generate --target dev
dbt docs serve
~~~

## 5. Deployment

The unified job runs dbt as defined in ingestion/bundle/bronze_ingestion_job.yml. From the repository root:

~~~bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run bronze_ingestion -t dev
~~~

The dbt_transformation task starts only after Bronze ingestion succeeds and runs notebooks/dbt_runner.py.

## 6. Structure

~~~text
dbt/
├── macros/                         # Reusable SQL rules
├── models/
│   ├── staging/                    # 24 Silver views
│   ├── intermediate/               # 13 Silver views
│   └── marts/                      # 15 Gold tables
├── notebooks/                      # Databricks dbt runner
├── tests/business_logic/           # Singular business logic tests
├── dbt_project.yml
├── packages.yml
├── package-lock.yml
├── profiles.yml.example
└── readme.md
~~~

The project produces 24 staging models, 13 intermediate models, 6 dimensions, 8 facts, and 1 daily channel aggregation.

## 7. Workflow

~~~mermaid
flowchart LR
    A[Bronze] --> B[Staging]
    B --> C[Intermediate]
    B --> D[Marts]
    C --> D
    D --> E[Tests]
~~~

The dbt_runner.py notebook installs dbt-databricks, locates dbt_project.yml, reads credentials from the retail_analytics Secret Scope, creates a temporary profile, and runs dbt deps, compile, run, and test.

dbt uses source() and ref() to build the dependency graph and determine model execution order.

## 8. Notes

- Never commit a profiles.yml containing real credentials.
- Run dbt deps after cloning the repository or changing packages.yml.
- Check logs/dbt.log for failure details.
- Check target/compiled to inspect generated SQL.
- Use --full-refresh only when a table must be recreated and the impact is understood.
- Confirm model grain before adding joins or aggregating metrics.
- CI runs SQLFluff, dbt parse, dbt compile, materializes dependencies for singular tests, and then runs those tests.
- Source ingestion is described in the [ingestion guide](../ingestion/readme.md).