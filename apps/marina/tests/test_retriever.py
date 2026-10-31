from __future__ import annotations

from unittest.mock import MagicMock, patch

from databricks.sdk.service.vectorsearch import (
    ColumnInfo,
    QueryVectorIndexResponse,
    ResultData,
    ResultManifest,
)

from apps.marina.src.retriever import (
    MIN_RELEVANCE_SCORE,
    RESULT_COLUMNS,
    VECTOR_INDEX,
    RetrievedChunk,
    has_sufficient_context,
    retrieval_query,
    retrieve,
    sources_for,
)


def _fake_response(rows):
    manifest = ResultManifest(
        column_count=len(RESULT_COLUMNS) + 1,
        columns=[ColumnInfo(name=name) for name in [*RESULT_COLUMNS, "score"]],
    )
    return QueryVectorIndexResponse(
        manifest=manifest,
        result=ResultData(data_array=rows, row_count=len(rows)),
    )


def test_retrieve_maps_columns_by_manifest_order():
    response = _fake_response(
        [["readme.md", "readme", "Título", "texto do chunk", 0.71]]
    )
    fake_client = MagicMock()
    fake_client.vector_search_indexes.query_index.return_value = response

    with patch("apps.marina.src.retriever.workspace_client", return_value=fake_client):
        chunks = retrieve("pergunta qualquer")

    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.source_path == "readme.md"
    assert chunk.source_type == "readme"
    assert chunk.heading == "Título"
    assert chunk.chunk_text == "texto do chunk"
    assert chunk.score == 0.71
    query_kwargs = fake_client.vector_search_indexes.query_index.call_args.kwargs
    assert query_kwargs["index_name"] == VECTOR_INDEX
    assert query_kwargs["score_threshold"] == MIN_RELEVANCE_SCORE


def test_retrieve_returns_empty_list_when_no_results():
    response = _fake_response([])
    fake_client = MagicMock()
    fake_client.vector_search_indexes.query_index.return_value = response

    with patch("apps.marina.src.retriever.workspace_client", return_value=fake_client):
        chunks = retrieve("pergunta sem contexto")

    assert chunks == []


def test_retrieve_caps_at_max_context_chunks():
    rows = [
        [f"file{i}.md", "readme", "", f"chunk {i}", 1.0 - i * 0.01] for i in range(6)
    ]
    response = _fake_response(rows)
    fake_client = MagicMock()
    fake_client.vector_search_indexes.query_index.return_value = response

    with patch("apps.marina.src.retriever.workspace_client", return_value=fake_client):
        chunks = retrieve("pergunta com muitos resultados")

    assert len(chunks) == 5


def test_has_sufficient_context_true_with_chunks():
    chunk = RetrievedChunk("a.md", "readme", "", "texto", 0.5)
    assert has_sufficient_context([chunk])


def test_has_sufficient_context_false_when_empty():
    assert not has_sufficient_context([])


def test_sources_for_deduplicates_preserving_order():
    chunks = [
        RetrievedChunk("a.md", "readme", "", "x", 0.9),
        RetrievedChunk("b.md", "readme", "", "y", 0.8),
        RetrievedChunk("a.md", "readme", "", "z", 0.7),
    ]
    assert sources_for(chunks) == ["a.md", "b.md"]


def test_retrieval_query_expands_forecast_rules():
    query = retrieval_query("Mostre as regras de cálculo do forecast.")

    assert "elegibilidade" in query
    assert "baseline fallback" in query
    assert "cutoff" in query


def test_retrieval_query_expands_feature_calculations():
    query = retrieval_query("Como são calculadas as features?")

    assert "int_forecast_feature_matrix" in query
    assert "lag_1_qty" in query
    assert "rolling_6m_avg_qty" in query


def test_retrieval_query_expands_defensive_security_guidance():
    query = retrieval_query("O que devo evitar para vazamento de dados?")

    assert "não versionar credenciais" in query
    assert "OAuth do service principal" in query
    assert "permissões mínimas" in query


def test_retrieval_query_preserves_unrelated_question():
    question = "Qual região é candidata a um novo CD?"

    assert retrieval_query(question) == question
