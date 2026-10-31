"""Testes do provisionamento do índice AI Search.

Usa dublês de teste simples em vez de `AISearchClient` real — os
testes cobrem a lógica de idempotência e de espera, não a integração
de rede com o Databricks (isso é validado manualmente/via execução real,
documentada no README).
"""

from __future__ import annotations

import pytest
from databricks.ai_search.exceptions import NotFound

from apps.marina.src.vector_index_setup import (
    ENDPOINT_NAME,
    INDEX_NAME,
    ensure_endpoint,
    ensure_index,
    sync_index,
    wait_until_ready,
)


class _FakeClient:
    def __init__(self, existing_endpoints=(), existing_index=None):
        self.existing_endpoints = list(existing_endpoints)
        self.existing_index = existing_index
        self.created_endpoint = False
        self.created_index = False

    def list_endpoints(self):
        return {"endpoints": [{"name": name} for name in self.existing_endpoints]}

    def create_endpoint(self, name, endpoint_type):
        self.created_endpoint = True
        self.existing_endpoints.append(name)

    def get_index(self, endpoint_name, index_name):
        if self.existing_index is None:
            raise NotFound("índice não existe")
        return self.existing_index

    def create_delta_sync_index(self, **kwargs):
        self.created_index = True
        return "novo-indice"


class _FakeIndex:
    def __init__(self, statuses):
        self._statuses = list(statuses)
        self.sync_called = False

    def describe(self):
        return {"status": self._statuses.pop(0)}

    def sync(self):
        self.sync_called = True


def test_ensure_endpoint_creates_when_missing():
    client = _FakeClient(existing_endpoints=[])
    ensure_endpoint(client)
    assert client.created_endpoint
    assert ENDPOINT_NAME in client.existing_endpoints


def test_ensure_endpoint_is_idempotent():
    client = _FakeClient(existing_endpoints=[ENDPOINT_NAME])
    ensure_endpoint(client)
    assert not client.created_endpoint


def test_ensure_index_reuses_existing_index():
    client = _FakeClient(existing_index="indice-existente")
    result = ensure_index(client)
    assert result == "indice-existente"
    assert not client.created_index


def test_ensure_index_creates_when_missing():
    client = _FakeClient(existing_index=None)
    result = ensure_index(client)
    assert result == "novo-indice"
    assert client.created_index


def test_ensure_index_propagates_unexpected_errors():
    class _FailingClient(_FakeClient):
        def get_index(self, endpoint_name, index_name):
            raise PermissionError("sem permissão")

    with pytest.raises(PermissionError):
        ensure_index(_FailingClient())


def test_sync_index_requests_a_sync():
    index = _FakeIndex([{"ready": True}])
    result = sync_index(_FakeClient(existing_index=index))
    assert result is index
    assert index.sync_called


def test_wait_until_ready_returns_when_ready():
    index = _FakeIndex([{"ready": False}, {"ready": True}])
    wait_until_ready(index, timeout_seconds=5, poll_seconds=0)


def test_wait_until_ready_raises_on_failure_state():
    index = _FakeIndex([{"ready": False, "detailed_state": "SYNC_FAILED"}])
    with pytest.raises(RuntimeError):
        wait_until_ready(index, timeout_seconds=5, poll_seconds=0)


def test_wait_until_ready_raises_on_timeout():
    index = _FakeIndex([{"ready": False}] * 100)
    with pytest.raises(TimeoutError):
        wait_until_ready(index, timeout_seconds=0, poll_seconds=0)


def test_index_name_matches_chunks_table_convention():
    assert INDEX_NAME.endswith("marina_document_chunks_idx")
