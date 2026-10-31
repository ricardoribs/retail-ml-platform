"""Shared Marina configuration: Databricks client and endpoints.

Endpoints are configurable through environment variables, following the
`MARINA_*` convention already used in `app.py` (`MARINA_CATALOG`,
`MARINA_MARTS_SCHEMA`, and others).
"""

from __future__ import annotations

import os

DEFAULT_LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
DEFAULT_MIN_RELEVANCE_SCORE = 0.5
DEFAULT_VECTOR_INDEX = "retail_analytics.default_marts.marina_document_chunks_idx"

LLM_ENDPOINT = os.environ.get("MARINA_LLM_ENDPOINT", DEFAULT_LLM_ENDPOINT)
VECTOR_INDEX = os.environ.get("MARINA_VECTOR_INDEX", DEFAULT_VECTOR_INDEX)
MIN_RELEVANCE_SCORE = float(
    os.environ.get("MARINA_MIN_RELEVANCE_SCORE", str(DEFAULT_MIN_RELEVANCE_SCORE))
)

if not 0 <= MIN_RELEVANCE_SCORE <= 1:
    raise ValueError("MARINA_MIN_RELEVANCE_SCORE must be between 0 and 1.")


def workspace_client():
    """Returns a Databricks client authenticated as the app service principal.

    Uses the same environment variables as `DatabricksMartProvider`
    (`app.py`): `DATABRICKS_CLIENT_ID`/`DATABRICKS_CLIENT_SECRET` and
    `DATABRICKS_SERVER_HOSTNAME`/`DATABRICKS_HOST`. Outside a Databricks
    App (local development), it falls back to the SDK's default
    authentication (local profile or generic Databricks CLI environment
    variables).
    """
    from databricks.sdk import WorkspaceClient

    client_id = os.environ.get("DATABRICKS_CLIENT_ID")
    client_secret = os.environ.get("DATABRICKS_CLIENT_SECRET")
    host = os.environ.get("DATABRICKS_SERVER_HOSTNAME") or os.environ.get(
        "DATABRICKS_HOST"
    )
    if client_id and client_secret and host:
        hostname = host if host.startswith("http") else f"https://{host}"
        return WorkspaceClient(
            host=hostname, client_id=client_id, client_secret=client_secret
        )
    return WorkspaceClient()
