"""Testes do indexador de documentos da Marina."""

from __future__ import annotations

from pathlib import Path

import pytest

from apps.marina.src.indexer import (
    REPO_ROOT,
    build_document_chunks,
    chunk_file,
    classify_source,
    discover_source_files,
    is_ignored,
    persist_chunks,
    split_markdown_sections,
    split_sql_blocks,
    split_yaml_sections,
)


def test_discover_source_files_finds_known_readmes():
    found = {p.relative_to(REPO_ROOT).as_posix() for p in discover_source_files()}
    assert "readme.md" in found
    assert "ml/demand_forecasting/README.md" in found
    assert "dbt/readme.md" in found


def test_discover_source_files_excludes_process_and_generated_sources():
    found = {p.relative_to(REPO_ROOT).as_posix() for p in discover_source_files()}
    assert "databricks.yml" not in found
    assert ".github/workflows/ci-cd.yml" not in found
    assert not any(path.startswith("ingestion/bundle/") for path in found)
    assert not any(path.startswith("specs/") for path in found)
    assert not any("/notebooks/" in ("/" + path + "/") for path in found)
    assert "apps/marina/src/prompts.py" not in found


def test_discover_source_files_never_returns_ignored_paths():
    for path in discover_source_files():
        assert not is_ignored(path)


def test_is_ignored_flags_secrets_and_build_artifacts():
    assert is_ignored(REPO_ROOT / ".env")
    assert is_ignored(REPO_ROOT / "dbt" / "profiles.yml")
    assert is_ignored(REPO_ROOT / "dbt" / "dbt_packages" / "dbt_utils" / "x.sql")
    assert is_ignored(REPO_ROOT / "dbt" / "target" / "compiled" / "x.sql")
    assert is_ignored(REPO_ROOT / "ml" / "__pycache__" / "x.pyc")
    assert is_ignored(REPO_ROOT / "specs" / "design.md")
    assert is_ignored(
        REPO_ROOT / "ml" / "demand_forecasting" / "notebooks" / "forecast.py"
    )
    assert is_ignored(REPO_ROOT / "secret.env")


def test_is_ignored_allows_real_source_file():
    assert not is_ignored(REPO_ROOT / "readme.md")


def test_classify_source_dbt_layers():
    assert classify_source(Path("dbt/models/staging/stg_orders.sql")) == (
        "sql_model",
        "staging",
    )
    assert classify_source(Path("dbt/models/intermediate/schema.yml")) == (
        "schema",
        "intermediate",
    )
    assert classify_source(Path("dbt/models/marts/fct_sales.sql")) == (
        "sql_model",
        "marts",
    )


def test_classify_source_repo_level_docs_and_config():
    assert classify_source(Path("readme.md")) == ("readme", "repo")
    assert classify_source(Path("apps/marina/src/indexer.py")) == ("code", "repo")


def test_split_markdown_sections_keeps_heading():
    text = "# Título\n\nintro\n\n## Sub\n\nconteúdo da sub"
    sections = split_markdown_sections(text)
    headings = [heading for heading, _ in sections]
    assert "Título" in headings
    assert "Sub" in headings


def test_split_markdown_sections_handles_preamble_without_heading():
    text = "texto solto antes de qualquer cabeçalho\n\n# Depois"
    sections = split_markdown_sections(text)
    assert sections[0][0] == ""
    assert "texto solto" in sections[0][1]


def test_split_markdown_sections_ignores_hash_inside_code_fence():
    text = (
        "# Título real\n\n"
        "texto normal\n\n"
        "```text\n"
        "pasta/\n"
        "# não é um cabeçalho, é um comentário dentro do bloco de código\n"
        "```\n\n"
        "## Depois do bloco\n\nconteúdo"
    )
    sections = split_markdown_sections(text)
    headings = [heading for heading, _ in sections]
    assert headings == ["Título real", "Depois do bloco"]


def test_split_sql_blocks_preserves_leading_comment_as_heading():
    text = (
        "-- Regra: usa apenas pedidos pagos\n"
        "-- Grão: order_id\n"
        "with base as (\n    select 1\n),\n"
        "final as (\n    select * from base\n)\n"
        "select * from final\n"
    )
    heading, _ = split_sql_blocks(text)[0]
    assert "Regra" in heading


def test_split_sql_blocks_without_comment_returns_whole_query():
    text = "select 1 as x"
    sections = split_sql_blocks(text)
    assert len(sections) == 1
    assert sections[0][1] == text


def test_split_yaml_sections_splits_top_level_keys():
    text = (
        "version: 2\n"
        "models:\n"
        "  - name: x\n"
        "    columns:\n"
        "      - name: y\n"
        "sources:\n"
        "  - name: z\n"
    )
    sections = split_yaml_sections(text)
    headings = [heading for heading, _ in sections]
    assert "version" in headings
    assert "models" in headings
    assert "sources" in headings


def test_chunk_file_produces_hashable_nonempty_chunks():
    chunks = chunk_file(REPO_ROOT / "ml" / "demand_forecasting" / "README.md")
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.chunk_text.strip()
        assert chunk.source_path == "ml/demand_forecasting/README.md"
        assert chunk.source_type == "readme"


def test_chunk_file_content_hash_is_deterministic():
    chunks_a = chunk_file(REPO_ROOT / "readme.md")
    chunks_b = chunk_file(REPO_ROOT / "readme.md")
    assert [c.content_hash for c in chunks_a] == [c.content_hash for c in chunks_b]
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]


def test_chunk_file_respects_target_size_range():
    chunks = chunk_file(REPO_ROOT / "ml" / "demand_forecasting" / "README.md")
    # Only the tail of a section may be much shorter than the target; most
    # chunks should stay within the upper bound of the design range.
    oversized = [c for c in chunks if len(c.chunk_text) > 1800]
    assert not oversized


def test_build_document_chunks_covers_every_discovered_file():
    files = discover_source_files()
    chunks = build_document_chunks()
    covered_paths = {c.source_path for c in chunks}
    expected_paths = {p.relative_to(REPO_ROOT).as_posix() for p in files}
    assert covered_paths == expected_paths


def test_chunk_ids_are_unique_across_whole_repo_scan():
    chunks = build_document_chunks()
    chunk_ids = [c.chunk_id for c in chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_persist_chunks_rejects_empty_scan_before_connecting_to_spark():
    with pytest.raises(ValueError, match="Nenhum chunk foi gerado"):
        persist_chunks([])
