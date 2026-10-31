"""Marina retriever: vector search over technical document chunks.

Uses the Vector Search index created in `vector_index_setup.py`
(`marina_document_chunks_idx`).
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from .config import MIN_RELEVANCE_SCORE, VECTOR_INDEX, workspace_client

TOP_K = 6
MAX_CONTEXT_CHUNKS = 5

RESULT_COLUMNS = ["source_path", "source_type", "heading", "chunk_text"]


def _normalized_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )


def retrieval_query(question: str) -> str:
    """Adds technical vocabulary to short or ambiguous questions."""
    normalized = _normalized_text(question)
    if any(
        term in normalized
        for term in (
            "security", "leak", "credential", "password", "authentication", "seguranca",
            "vazamento",
            "credencial",
            "token",
            "senha",
            "autenticacao",
        )
    ):
        return (
            f"{question}\n"
            "Documented security controls: never commit credentials or "
            "tokens; use service principal OAuth and valueFrom resources in the "
            "Databricks App; apply least privilege; do not index "
            "sensitive files or expose values in answers or logs."
        )
    if "forecast" in normalized or "previs" in normalized:
        if any(term in normalized for term in ("rules", "rule", "calculation", "how does", "how it works", "regra", "calculo", "como funciona")):
            return (
                f"{question}\n"
                "Explicit demand forecasting rules: eligibility, "
                "regression, baseline fallback, cutoff, horizon, models, and metrics."
            )
    if "feature" in normalized or "atributo" in normalized:
        return (
            f"{question}\n"
            "Calculations implemented in int_forecast_feature_matrix: demand_qty, "
            "lag_1_qty, lag_2_qty, lag_3_qty, rolling_3m_avg_qty, "
            "rolling_6m_avg_qty, and rolling_3m_sum_qty."
        )
    return question


@dataclass(frozen=True)
class RetrievedChunk:
    source_path: str
    source_type: str
    heading: str
    chunk_text: str
    score: float


def retrieve(question: str, num_results: int = TOP_K) -> list[RetrievedChunk]:
    """Retrieves the chunks most relevant to the question.

    Returns at most `MAX_CONTEXT_CHUNKS`. The index may return up to
    `num_results`, but only the first results are included in the LLM context (design,
    section 9.5: top_k=6 in search, max_context_chunks=5 in the final prompt).
    """
    client = workspace_client()
    response = client.vector_search_indexes.query_index(
        index_name=VECTOR_INDEX,
        columns=RESULT_COLUMNS,
        query_text=retrieval_query(question),
        num_results=num_results,
        score_threshold=MIN_RELEVANCE_SCORE,
    )
    if response.result is None or not response.result.data_array:
        return []

    manifest_columns = (
        [column.name for column in response.manifest.columns]
        if response.manifest and response.manifest.columns
        else [*RESULT_COLUMNS, "score"]
    )

    chunks = []
    for row in response.result.data_array:
        record = dict(zip(manifest_columns, row))
        chunks.append(
            RetrievedChunk(
                source_path=record["source_path"],
                source_type=record["source_type"],
                heading=record.get("heading") or "",
                chunk_text=record["chunk_text"],
                score=float(record.get("score") or 0.0),
            )
        )
    return chunks[:MAX_CONTEXT_CHUNKS]


def has_sufficient_context(chunks: list[RetrievedChunk]) -> bool:
    """Confirms that search returned at least one chunk above the threshold."""
    return len(chunks) > 0


def sources_for(chunks: list[RetrievedChunk]) -> list[str]:
    """Return unique `source_path` values in relevance order."""
    seen: list[str] = []
    for chunk in chunks:
        if chunk.source_path not in seen:
            seen.append(chunk.source_path)
    return seen
