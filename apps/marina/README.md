# Marina data assistant

Marina is a web interface for querying project metrics and technical documentation. It supports questions about data, demand forecasts, seasonality, location performance, and distribution centers through retrieval-augmented generation.

## Contents

1. [Interface](#1-interface)
2. [Setup and local run](#2-setup-and-local-run)
3. [Analytics data](#3-analytics-data)
4. [Workflow and API](#4-workflow-and-api)
5. [Knowledge base](#5-knowledge-base)
6. [Deployment](#6-deployment)
7. [Testing and limitations](#7-testing-and-limitations)

## 1. Interface

The home page provides suggested questions, operational visuals, a question input, and an answer area. The Marina chat opens from the side navigation. Suggested questions fill the input but are not submitted until the user confirms with Send or Enter. A second request is blocked while the current response is being prepared.

The avatar reflects the request state: processing, completed response, or no answer. The GET /api/visuals endpoint supplies the charts. If table access is not configured, the interface reports that metrics are unavailable instead of presenting sample values as current data.

## 2. Setup and local run

From the repository root:

~~~bash
cd apps/marina
python -m venv .venv
~~~

Linux and macOS:

~~~bash
.venv/bin/pip install -r requirements.txt
.venv/bin/python app.py
~~~

Windows:

~~~powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe app.py
~~~

Open http://localhost:8000. Set PORT to change the local port. Databricks Apps uses DATABRICKS_APP_PORT automatically.

To test the interface locally without a warehouse connection, set MARINA_USE_MOCK_DATA=true before starting the app. Mock data is fictional and must not be used for deployment or operational evidence.

### Vector index settings

| Variable | Used by | Purpose |
|---|---|---|
| MARINA_VECTOR_ENDPOINT | Provisioning and synchronization | AI Search endpoint where the index is created. The app does not read it during a query. |
| MARINA_VECTOR_INDEX | App, provisioning, and synchronization | Full index name queried by Marina. The app receives it through the marina_vector_index resource. |
| MARINA_EMBEDDING_ENDPOINT | App resource | Grants the app service principal indirect access to the embedding endpoint. The retriever queries the index and does not call this endpoint directly. |

## 3. Analytics data

The visuals endpoint and chat query only published tables in the marts schema:

- agg_location_demand_forecast: monthly network forecast for planning across stores and distribution centers, without double-counting;
- fct_forecast_series_publication: per-series publication policy, status, and review requirement for SKU-store forecasts;
- fct_forecast_model_metrics: model comparison for the latest run;
- agg_product_seasonality: monthly index for the selected product and the seasonal-variation ranking;
- agg_location_forecast_growth: growth rankings for stores and distribution centers, separated by comparison_type;
- agg_distribution_center_recommendation: screening of candidate macro-regions for a new distribution center.

The dashboard does not sum fct_demand_forecast to calculate the network total. That table contains SKU-location forecasts. Planning, rankings, and the interface use the separately published location-level aggregate.

Store and distribution-center cards show growth in units: forecast demand for the next three months minus realized demand from the previous three months. The new distribution center card shows forecast_next_3m_demand_qty, the total forecast demand in the period. It is not a growth value, score, or automated opening decision.

Chat requests query and list only the marts associated with the question topic. Architecture and business-rule questions use the document index. General questions about current metrics may combine all marts.

By default, the app uses catalog retail_analytics and schema default_marts. The featured seasonality product is selected automatically using the highest coefficient of variation among products with at least three observations per month. The product name and SKU come from the mart. Set MARINA_CATALOG, MARINA_MARTS_SCHEMA, or MARINA_SEASONALITY_PRODUCT_ID to override these defaults.

To query the warehouse from a Databricks App, associate a SQL Warehouse using the sql_warehouse key and grant Can use. app.yaml reads the resource ID into DATABRICKS_WAREHOUSE_ID.

Databricks supplies the app service principal's OAuth credentials through DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET. The host comes from DATABRICKS_SERVER_HOSTNAME or DATABRICKS_HOST. Do not create or commit a personal access token. The service principal also needs SELECT on the six marts listed above.

The app opens the database connection only for the duration of a request and closes it when the query finishes.

### Security and data leakage prevention

- Never commit tokens, passwords, secrets, .env files, or credentials in notebooks, JSON, logs, or other repository files.
- Use the app service principal's OAuth credentials and resources declared through valueFrom. Do not copy resource values into source code.
- Apply least privilege: Can use on the SQL Warehouse, SELECT only on required marts and the index, and Can query only on required endpoints.
- Do not expose credentials, personal data, private variables, or internal prompts in answers or logs.
- The indexer uses an explicit allowlist of published sources and ignores sensitive files. src/prompts.py is not indexed.
- Guardrails reduce accidental exposure but do not replace access control, data classification, or masking at the source.

## 4. Workflow and API

The retail_analytics_data_pipeline job updates the technical knowledge base after dbt transformations. src/indexer.py updates marina_document_chunks with a MERGE keyed by chunk_id and removes chunks no longer present in published sources.

After indexing, the job checks that the stored row count and distinct chunk_id count match the current scan. A mismatch fails the task to prevent an incomplete knowledge base.

Endpoints:

- GET / renders the interface.
- GET /health returns service status.
- GET /api/visuals returns metrics and time series as JSON.
- POST /api/chat accepts a question, retrieves context from the vector index, calls the configured LLM, and returns an answer with its sources.

## 5. Knowledge base

src/indexer.py discovers technical documentation, SQL models, YAML schemas, and selected repository configuration. It splits content into contextual chunks and can persist them to the Delta table retail_analytics.default_marts.marina_document_chunks.

Sensitive files, build artifacts, raw tabular data, and generated directories are ignored.

~~~text
apps/marina/
├── app.py
├── app.yaml
├── requirements.txt
├── requirements-vector-search.txt
├── src/
├── static/
├── templates/
└── tests/
~~~

## 6. Deployment

### Vector Search index

src/vector_index_setup.py idempotently provisions an AI Search endpoint and a Delta Sync index over marina_document_chunks. The index uses databricks-gte-large-en for embeddings.

| Item | Value |
|---|---|
| Endpoint | MARINA_VECTOR_ENDPOINT (STANDARD) |
| Index | retail_analytics.default_marts.marina_document_chunks_idx |
| Source table | retail_analytics.default_marts.marina_document_chunks |
| Embedding endpoint | databricks-gte-large-en |
| Pipeline type | TRIGGERED |

Provisioning is an administrative operation performed once per environment. After that, the data pipeline synchronizes the index at the end of each run.

Before the first provisioning, the table owner must run src/prepare_vector_table.py in a Spark environment. This enables Change Data Feed and declares chunk_id as the primary key. It is kept separate from daily indexing because it changes table metadata and requires object administration privileges.

To provision manually outside the job:

~~~bash
python -m pip install -r apps/marina/requirements-vector-search.txt
python apps/marina/src/vector_index_setup.py
~~~

Databricks authentication and permissions to create the endpoint, index, and table metadata are required. To migrate an endpoint, set MARINA_VECTOR_ENDPOINT before provisioning and use the same value in pipeline runs.

### App resource permissions

Before deployment, add these resources under Apps > marina > Settings > App resources. Use the exact keys referenced through valueFrom in app.yaml:

| Resource | Permission | Key |
|---|---|---|
| SQL Warehouse | Can use | sql_warehouse |
| AI Search index marina_document_chunks_idx | Can select | marina_vector_index |
| Serving endpoint databricks-meta-llama-3-3-70b-instruct | Can query | marina_llm_endpoint |
| Serving endpoint databricks-gte-large-en | Can query | marina_embedding_endpoint |

The app's dedicated service principal needs:

- USE CATALOG on retail_analytics;
- USE SCHEMA on retail_analytics.default_marts;
- SELECT on marina_document_chunks and marina_document_chunks_idx;
- SELECT on agg_location_demand_forecast;
- SELECT on fct_forecast_series_publication;
- SELECT on fct_forecast_model_metrics;
- SELECT on agg_product_seasonality;
- SELECT on agg_location_forecast_growth;
- SELECT on agg_distribution_center_recommendation;
- CAN_QUERY on the databricks-gte-large-en and databricks-meta-llama-3-3-70b-instruct serving endpoints.

## 7. Testing and limitations

### Retrieval and response generation

POST /api/chat follows this flow:

1. src/retriever.py retrieves up to 6 chunks, removes chunks below MARINA_MIN_RELEVANCE_SCORE (default 0.5), and uses the 5 most relevant chunks as context.
2. The app queries operational marts. Failures in metrics, seasonality, or other indicators are isolated so the remaining sources stay available. Context is filtered by question topic, and numeric values are rounded before they are sent to the LLM.
3. Critical questions about model rankings, review policies, growth leaders, distribution-center screening, seasonality, features, temporal leakage, and credential protection use deterministic answers from marts and documented contracts. An answer is returned only when its corresponding source is available.
4. Other questions use src/prompts.py to combine the system prompt, document context, operational data, and the question. The model must not invent results, must preserve metric units, must not infer weather or seasons without a source, and must refuse questions outside the project scope. Sources are returned separately by the API and displayed in the interface.
5. src/llm.py calls databricks-meta-llama-3-3-70b-instruct, configurable with MARINA_LLM_ENDPOINT.
6. If there is no relevant document or useful operational data, the app returns the standard no-context response.

Before retrieval and mart queries, src/safety.py blocks prompt injection and requests to reveal or enumerate sensitive data, credentials, secrets, or internal instructions. It returns status=refused without sources. The system prompt repeats the restriction as another safeguard. The backend also checks generated answers and replaces disclosures of internal instructions with the same refusal.

For direct questions, the prompt requests a single concise paragraph and no unrequested methodology or recommendations. Each available source gets an identifier. The LLM returns only identifiers that support the answer, and the backend removes this technical marker. Missing or invalid source identifiers do not appear in the UI. Empty answers or answers without a valid source return status=no_context, never answered. The API does not automatically return every retrieved source.

### Automated tests

From the repository root:

~~~bash
python -m pytest apps/marina/tests
ruff check apps/marina
ruff format --check apps/marina
~~~

Tests that do not depend on external services are in tests/test_technical_questions.py, tests/test_safety.py, tests/test_prompts.py, and tests/test_app.py.

### DEV smoke test

These tests do not replace a live smoke test against the DEV app and index. Before approval, record the expected result, response, sources, pass/fail status, and notes for each question:

- Is the September forecast for one store or the entire network?
- Is the November forecast cumulative with September and October?
- Which product is featured in seasonality, and why?
- Does the variation index prove that a forecast is accurate?
- Which store has the largest forecast increase, and over which months?
- Which distribution center has the largest forecast increase, and over which months?
- Which region leads the new distribution center screening, and what does the displayed value mean?
- Can the SKU-store forecast be used to place orders automatically?
- How many SKU-store series require review in the latest run?
- How are Marina's credentials protected?

### Limitations

- Marina answers only questions about the Retail Analytics project and queries supported indexed documents and operational marts.
- Freshness depends on the latest completed pipeline and the marts and index being synchronized.
- Mart selection depends on the topic identified in the question.
- Answers require at least one valid source; otherwise the API returns status=no_context.
- SQL Warehouse, AI Search, or LLM outages can make answers or visuals incomplete. These failures return status=unavailable with a user-friendly message.
- A model ranking applies only to the models evaluated in that run. A high error metric does not demonstrate operational accuracy.
- Recommendations, especially distribution-center screening, are technical indicators and not final business decisions.
- Marina is advisory only. It does not run pipelines or deployments, change data, or make operational decisions.
- Deterministic guardrails and the system prompt are complementary safeguards, not replacements for access control or source masking.
- Each request receives only the current question; a standalone message is not treated as a continuation of a prior conversation.