"""Provision Marina's Vector Search index.

Creates the endpoint and Delta Sync index on `marina_document_chunks`
idempotently, using the `databricks-gte-large-en` embedding endpoint.

The source table prerequisites, applied by `prepare_vector_table.py`, are
Change Data Feed enabled and `chunk_id` set as the primary key, as required
by the Delta Sync index.

Runs outside a notebook or job. It uses `AISearchClient` with Databricks
credentials available in the environment and does not depend on
`spark` or `dbutils`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _resolve_repo_root() -> Path:
    """Find the repository root when this file is run as a script."""
    try:
        start = Path(__file__).resolve()
    except NameError:
        start = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else Path.cwd()
    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start


REPO_ROOT = _resolve_repo_root()
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.demand_forecasting.src.config import MARTS_SCHEMA, fq_table  # noqa: E402

ENDPOINT_NAME = os.getenv("MARINA_VECTOR_ENDPOINT", "marina-cp5-vector-search")
SOURCE_TABLE = fq_table(MARTS_SCHEMA, "marina_document_chunks")
INDEX_NAME = os.getenv(
    "MARINA_VECTOR_INDEX",
    fq_table(MARTS_SCHEMA, "marina_document_chunks_idx"),
)
EMBEDDING_MODEL_ENDPOINT = "databricks-gte-large-en"


def create_client():
    """Create the Databricks AI Search client using the active credentials."""
    from databricks.ai_search.client import AISearchClient

    return AISearchClient(disable_notice=True)


def ensure_endpoint(client) -> None:
    """Creates the STANDARD endpoint if it does not already exist."""
    existing = {ep["name"] for ep in client.list_endpoints().get("endpoints", [])}
    if ENDPOINT_NAME not in existing:
        client.create_endpoint(name=ENDPOINT_NAME, endpoint_type="STANDARD")


def ensure_index(client):
    """Creates the Delta Sync index if needed and returns its handle."""
    try:
        return client.get_index(endpoint_name=ENDPOINT_NAME, index_name=INDEX_NAME)
    except Exception as exc:
        from databricks.ai_search.exceptions import NotFound

        if not isinstance(exc, NotFound):
            raise
        return client.create_delta_sync_index(
            endpoint_name=ENDPOINT_NAME,
            source_table_name=SOURCE_TABLE,
            index_name=INDEX_NAME,
            pipeline_type="TRIGGERED",
            primary_key="chunk_id",
            embedding_source_column="chunk_text",
            embedding_model_endpoint_name=EMBEDDING_MODEL_ENDPOINT,
        )


def sync_index(client):
    """Starts an index sync and returns its handle."""
    index = client.get_index(endpoint_name=ENDPOINT_NAME, index_name=INDEX_NAME)
    index.sync()
    return index


def wait_until_ready(
    index, timeout_seconds: int = 1800, poll_seconds: int = 20
) -> None:
    """Waits until the index is queryable or the timeout expires."""
    import time

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        status = index.describe().get("status", {})
        if status.get("ready"):
            return
        if status.get("detailed_state") in ("PROVISIONING_FAILED", "SYNC_FAILED"):
            raise RuntimeError(f"Index provisioning failed: {status}")
        time.sleep(poll_seconds)
    raise TimeoutError(f"Index was not ready within {timeout_seconds}s")


def main() -> None:
    client = create_client()
    ensure_endpoint(client)
    index = ensure_index(client)
    print(f"Waiting for index to become ready: {INDEX_NAME}")
    wait_until_ready(index)
    print(f"Index ready: {INDEX_NAME} (endpoint: {ENDPOINT_NAME})")


if __name__ == "__main__":
    main()
