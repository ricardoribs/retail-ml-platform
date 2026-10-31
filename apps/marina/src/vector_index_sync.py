"""Sync Marina's search index after the Delta table is updated."""

from __future__ import annotations

from vector_index_setup import (
    ENDPOINT_NAME,
    INDEX_NAME,
    create_client,
    sync_index,
    wait_until_ready,
)


def main() -> None:
    client = create_client()
    index = sync_index(client)
    print(f"Syncing index: {INDEX_NAME}")
    wait_until_ready(index)
    print(f"Index synced: {INDEX_NAME} (endpoint: {ENDPOINT_NAME})")


if __name__ == "__main__":
    main()
