# Retail ML Platform — Machine Learning and Data Science

This repository contains the stages of a data and ML platform for demand forecasting and inventory planning. Its main focus is the Data Science workflow: data audits and preparation, feature engineering, model selection and backtesting, forecast publication, and results analysis.

Ingestion, dbt, and Databricks provide the data foundation. Power BI and the query assistant are consumption layers. Together, these components demonstrate the path from transactional sources to forecast evaluation and operational use.

## Contents

1. [Architecture](#1-architecture)
2. [Integrations](#2-integrations)
3. [Execution flow](#3-execution-flow)
4. [Configuration](#4-configuration)
5. [Running the project](#5-running-the-project)
6. [Implementation status](#6-implementation-status)
7. [Documentation](#7-documentation)

## 1. Architecture

The ML workflow uses the Bronze, Silver, and Gold layers for data preparation. The project's core is ml/demand_forecasting/, which contains audits, feature engineering, baselines, model selection, backtests, publication rules, and forecasts at different granularities.

### ML and Data Science focus

The forecasting module includes:

- Audits of time series, seasonality, availability, and intermittent demand.
- Time-aware feature construction to prevent data leakage.
- Backtesting to compare baselines and models.
- Product-location forecasting, hierarchical reconciliation, and allocation.
- Eligibility, review, and publication policies.

Notebooks retain the experiment history, while src/ contains reusable components and unit tests.

~~~mermaid
flowchart LR
    subgraph foundation[Data foundation]
        source[Transactional sources] --> bronze[Bronze ingestion]
        bronze --> silver[dbt modeling: Silver and Gold]
    end

    subgraph ml[ML and Data Science workflow]
        audit[Data audit and feature matrix]
        models[Model selection and backtesting]
        forecast[Product-location forecasts]
        planning[Reconciliation, allocation, and planning]
        audit --> models --> forecast --> planning
    end

    silver --> audit
    silver --> powerbi[Power BI: Gold layer analytics]
    silver --> assistant[Data assistant with RAG]
    planning --> assistant
~~~

Ingestion and transformation provide the organized data needed to reproduce the experiments. The forecasting module contains the repository's main ML and Data Science components.

### Bronze

The Bronze layer extracts 24 SQL Server tables and stores them in Delta Lake. Ingestion supports full and incremental loads, idempotent operations, and watermark tracking.

See the [ingestion guide](./ingestion/readme.md) for connection details, load strategies, control tables, notebooks, and monitoring.

### Silver and Gold

The dbt project reads from Bronze, standardizes the data, applies business rules, and builds the Gold analytical model. It contains 24 staging models, 13 intermediate models, 6 dimensions, 8 facts, 1 daily channel aggregation, and generic and singular data quality tests.

See the [dbt guide](./dbt/readme.md) for model architecture, macros, tests, configuration, and selective commands.

### Analytics and consumption

The Power BI report reads the Gold layer in retail_analytics.default_marts through the native Databricks connector and a SQL Warehouse. It imports 12 tables and includes 112 DAX measures across 7 analysis pages.

See the [Power BI guide](./powerbi/readme.md) for connection settings, refresh steps, the semantic model, and report pages.

### Main technologies

| Area | Technology | Purpose |
|---|---|---|
| ML/DS | Python and Databricks ML | Audits, features, model selection, backtests, and demand forecasts. |
| Platform | Databricks | Compute, storage, and workflows. |
| Source | SQL Server | Transactional data source for the runtime environment. |
| Processing | Apache Spark and PySpark | Extraction and distributed processing. |
| Storage | Delta Lake | Transactional tables and merge operations. |
| Transformation | dbt Core | SQL modeling, tests, documentation, and lineage. |
| Orchestration | Databricks Workflows | Scheduling, dependencies, retries, and notifications. |
| Governance | Unity Catalog | Organization and access control. |
| CI/CD | GitHub Actions | Linting, dbt validation, Databricks job execution, and release tags. |
| Analytics | Power BI | Semantic model and reports over the Gold layer. |
| Data assistant | Databricks Vector Search | Natural language queries with retrieval-augmented generation (RAG). |

## 2. Integrations

### Demand forecasting

Models in ml/demand_forecasting/ publish two complementary outputs: SKU-location forecasts governed by per-series review policies, and location-level aggregates for capacity planning, rankings, and distribution-center screening.

The [forecasting guide](./ml/demand_forecasting/README.md) describes the data, features, models, metrics, publication contracts, and known limitations.

### SQL Server and Databricks

SQL Server provides 24 operational tables. Connections use JDBC and credentials stored in a Databricks Secret Scope. Databricks hosts Delta tables, runs notebooks, and coordinates the retail_analytics_data_pipeline workflow through Databricks Asset Bundles.

### dbt

The dbt-databricks adapter runs Silver and Gold transformations in Databricks. Model dependencies are declared with source() and ref().

### GitHub Actions

The [CI/CD workflow](./.github/workflows/ci-cd.yml) validates pull requests and handles deployments:

- Pull requests validate branch naming, commits, and titles; Ruff and Python tests run when relevant files change.
- Branches named feat/*, feature/*, and hotfix/* can target main directly or go through dev. Pull requests from dev to main are also allowed.
- dbt changes trigger SQLFluff, dbt parse, dbt compile, dependency materialization, and selected tests. Workflow logs and dbt artifacts remain available in each run.
- On pull requests from forks, jobs that require Databricks credentials are skipped because GitHub does not provide repository secrets to those workflows.
- A push to dev validates and deploys the bundle to development, then starts the Databricks job.
- A push to main does not deploy. To deploy to production, open Actions > CI/CD > Run workflow, select main and the production environment, and run the workflow. Environment approval is required before the job runs and creates a semantic version tag after success.

Set these values under Settings > Secrets and variables > Actions:

| Type | Name |
|---|---|
| Variable | DATABRICKS_HOST |
| Variable | DATABRICKS_HTTP_PATH |
| Variable | DATABRICKS_JOB_ID |
| Secret | DATABRICKS_CLIENT_ID |
| Secret | DATABRICKS_CLIENT_SECRET |

Create the GitHub environments development and production. Add required reviewers to production so deployments wait for approval.

For dbt pull request validation, also configure Databricks variables and secrets at repository scope. If environments use different workspaces or jobs, define the same names in the corresponding environment to override repository-level values during deployment.

### Power BI and Marina

Power BI reads Gold tables through the native Databricks connector and authenticates with a Personal Access Token. Refresh the report after the daily pipeline completes.

The application in apps/marina/ answers natural language questions about Gold-layer data using Databricks Vector Search and RAG.

## 3. Execution flow

The reference workflow is defined in ingestion/bundle/bronze_ingestion_job.yml. It prepares data and runs the analytical stages in Databricks. Configure the schedule for each environment.

~~~mermaid
flowchart LR
    setup[setup_bronze] --> bronze[bronze_ingestion] --> dbt[dbt_transformation]

    dbt --> train[demand_forecast_train_evaluate]
    train --> segment[forecast_segment_backtest]
    train --> selection[forecast_series_model_selection]
    dbt --> seasonality[seasonality_ranking] --> policy[forecast_series_policy]

    train --> generate[demand_forecast_generate]
    segment --> generate
    selection --> generate
    policy --> generate

    generate --> replenishment[replenishment_planning_policy]
    generate --> aggregate[location_aggregate_forecast]
    aggregate --> gate[forecast_actionability_gate]
    gate --> growth[location_forecast_growth]
    growth --> dc_growth[cd_forecast_growth]
    dc_growth --> recommendation[distribution_center_recommendation]
    recommendation --> index[marina_document_indexer]
    index --> sync[marina_vector_index_sync]
~~~

1. setup_bronze validates configuration, the control table, and the SQL Server connection.
2. bronze_ingestion refreshes the Bronze tables.
3. dbt_transformation installs dependencies, compiles, runs, and tests the dbt models.
4. Forecast tasks evaluate SKU-location series, publish review policies, and generate granular forecasts.
5. Location-level aggregates feed rankings and distribution-center screening, then update Marina's document index.

Each task starts after its declared bundle dependencies. Forecast, publication, and indicator tasks can also run separately for experimentation. The job limits concurrent runs and configures retries per task. The Power BI report refresh happens after the pipeline completes and is not part of the job.

For task-level details, see the [Bronze ingestion guide](./ingestion/readme.md), [dbt guide](./dbt/readme.md), and [Power BI guide](./powerbi/readme.md#3-refreshing-the-data).

## 4. Configuration

### Prerequisites

- A Databricks workspace with Unity Catalog.
- Databricks CLI.
- Python 3.12 or a compatible version.
- Access to the source SQL Server.
- A SQL Warehouse for dbt.
- Read and write permissions on the required schemas.

Ingestion and dbt credentials are stored in separate Secret Scopes. Never commit tokens or passwords. See the [ingestion guide](./ingestion/readme.md#1-configure-ingestion), [dbt guide](./dbt/readme.md#1-set-up-dbt), and [Power BI guide](./powerbi/readme.md#2-connect-to-databricks).

The root databricks.yml defines dev and prod targets without hard-coding a workspace, identity, or credentials. Configure a local Databricks profile or environment variables, and review resources before deployment.

## 5. Running the project

From the repository root:

~~~bash
databricks bundle validate -t dev
databricks bundle deploy -t dev
databricks bundle run bronze_ingestion -t dev
~~~

For production, replace dev with prod. To run dbt separately, go to dbt and run:

~~~bash
dbt deps
dbt build --target dev
~~~

After the daily pipeline completes and the SQL Warehouse is active, open the report in Power BI Desktop and select Home > Refresh.

Monitor tasks, retries, and logs in Databricks Workflows. Check retail_analytics.bronze.ingestion_control for load status and dbt/logs/dbt.log for transformation details.

## 6. Implementation status

Delivered components include ingestion of 24 tables into Bronze; incremental loads, watermark tracking, and a control table; a unified Databricks job; tested Silver and Gold layers; a dimensional model; GitHub Actions CI/CD; a Power BI report with 7 pages and 112 DAX measures; module documentation; demand forecasting; and the Marina RAG assistant.

## 7. Documentation

- [Ingestion guide](./ingestion/readme.md): connections, configuration, load strategies, execution, and Bronze monitoring.
- [dbt guide](./dbt/readme.md): Silver and Gold architecture, models, macros, tests, configuration, and execution.
- [Power BI guide](./powerbi/readme.md): report access, Databricks connection, refresh, data architecture, semantic model, and pages.
- [Marina AI](./apps/marina/README.md)
- [Demand forecasting](./ml/demand_forecasting/README.md)

## Repository structure

~~~text
retail-ml-platform/
├── ingestion/                 # Bronze ingestion and Databricks bundle
├── dbt/                       # Silver and Gold transformations
├── powerbi/                   # Power BI documentation
├── apps/marina/               # Natural language data assistant
├── ml/demand_forecasting/     # Forecasting models and experiments
├── data-quality-gate-pilot/   # Isolated dbt quality gate pilot
├── docs/                      # Technical documentation
├── .github/workflows/         # CI/CD workflows
├── databricks.yml             # Root Databricks Asset Bundle
├── pyproject.toml             # Ruff and pytest configuration
└── readme.md                  # Project overview
~~~