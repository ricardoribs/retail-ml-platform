"""Index Marina documents for retrieval-augmented generation (RAG).

Reads the technical sources published in the repository, splits their
content into hierarchical chunks, and stores the chunks in a Delta table
for use by a vector index.

Only source discovery and chunking depend on the standard library (and can
be tested without PySpark). `persist_chunks` is the only function that must
run on a cluster with `spark` and Delta available.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _resolve_repo_root() -> Path:
    """Finds the repository root without relying on `__file__`.

    A Databricks Jobs `spark_python_task` executes this file through
    `exec()`, where `__file__` does not exist (unlike normal module imports
    in notebooks or tests). The search therefore uses `sys.argv[0]` as a
    fallback and walks up the directory tree until it finds the root
    `pyproject.toml`. This works in both contexts.
    """
    try:
        start = Path(__file__).resolve()
    except NameError:
        start = Path(sys.argv[0]).resolve() if sys.argv and sys.argv[0] else Path.cwd()

    for candidate in (start, *start.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return start


REPO_ROOT = _resolve_repo_root()

# Databricks Jobs also runs this file directly. In that case, the script
# directory may be `apps/marina/src` instead of the repository root.
# Adding the root to the path keeps shared imports working in both modes.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.demand_forecasting.src.config import MARTS_SCHEMA, fq_table  # noqa: E402

CHUNKS_TABLE = fq_table(MARTS_SCHEMA, "marina_document_chunks")

# Technical sources published for indexing.
SOURCE_GLOBS: tuple[str, ...] = (
    "readme.md",
    "ingestion/readme.md",
    "dbt/readme.md",
    "powerbi/readme.md",
    "dbt/models/staging/**/*.sql",
    "dbt/models/intermediate/**/*.sql",
    "dbt/models/marts/**/*.sql",
    "dbt/models/staging/**/*.yml",
    "dbt/models/intermediate/**/*.yml",
    "dbt/models/marts/**/*.yml",
    "ml/demand_forecasting/README.md",
    "apps/marina/README.md",
)

# Additional defense beyond the allowlist above.
IGNORED_PATH_PARTS: frozenset[str] = frozenset(
    {
        ".git",
        ".databricks",
        "dbt_packages",
        "target",
        "__pycache__",
        "specs",
        "notebooks",
    }
)
IGNORED_FILENAMES: frozenset[str] = frozenset({"profiles.yml", ".env"})
IGNORED_SUFFIXES: frozenset[str] = frozenset(
    {".parquet", ".csv", ".xlsx", ".pbix", ".env"}
)
IGNORED_KEYWORDS: tuple[str, ...] = ("token", "log", "cache")

# Hierarchical chunking (design section 9.4): 1,200-1,800 characters,
# with 150-250 characters of overlap.
TARGET_CHUNK_SIZE = 1500
CHUNK_OVERLAP = 200


def is_ignored(path: Path, repo_root: Path = REPO_ROOT) -> bool:
    """Defensive filter: never index secrets, binaries, or artifacts."""
    relative = path.relative_to(repo_root)
    parts_lower = [part.lower() for part in relative.parts]

    if any(part in IGNORED_PATH_PARTS for part in parts_lower):
        return True
    if path.name.lower() in IGNORED_FILENAMES:
        return True
    if path.suffix.lower() in IGNORED_SUFFIXES:
        return True
    if any(keyword in part for part in parts_lower for keyword in IGNORED_KEYWORDS):
        return True
    return False


def discover_source_files(repo_root: Path = REPO_ROOT) -> list[Path]:
    """Resolve `SOURCE_GLOBS` em arquivos reais, aplicando `is_ignored`."""
    matched: list[Path] = []
    for pattern in SOURCE_GLOBS:
        for path in sorted(repo_root.glob(pattern)):
            if path.is_file() and not is_ignored(path, repo_root):
                matched.append(path)
    return matched


def classify_source(relative_path: Path) -> tuple[str, str]:
    """Derives `source_type` and `source_layer` from the file path."""
    parts = relative_path.parts
    suffix = relative_path.suffix.lower()

    if suffix == ".md":
        source_type = "readme"
    elif suffix == ".sql":
        source_type = "sql_model"
    elif suffix in (".yml", ".yaml"):
        source_type = (
            "schema" if ("dbt" in parts and "models" in parts) else "pipeline_config"
        )
    else:
        source_type = "code"

    if "staging" in parts:
        source_layer = "staging"
    elif "intermediate" in parts:
        source_layer = "intermediate"
    elif "marts" in parts:
        source_layer = "marts"
    else:
        source_layer = "repo"

    return source_type, source_layer


def object_name_for(relative_path: Path) -> str:
    """Returns the logical object name: a dbt model name or relative path."""
    if relative_path.suffix.lower() in (".sql", ".yml", ".yaml") and (
        "models" in relative_path.parts
    ):
        return relative_path.stem
    return relative_path.as_posix()


def _split_by_size(
    text: str, target_size: int = TARGET_CHUNK_SIZE, overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split a text block while respecting the maximum chunk size.

    Splits only on spaces, never in the middle of a word or technical
    identifier (design: "do not split technical names such as
    product_variant_id...").
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= target_size:
        return [text]

    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + target_size, length)
        if end < length:
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks


_MARKDOWN_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)
_CODE_FENCE_RE = re.compile(r"^```.*$", re.MULTILINE)


def _code_fence_spans(text: str) -> list[tuple[int, int]]:
    """Return the (start, end) offsets of each fenced code block in the text."""
    fence_positions = [match.start() for match in _CODE_FENCE_RE.finditer(text)]
    return list(zip(fence_positions[0::2], fence_positions[1::2]))


def _is_inside_any_span(position: int, spans: list[tuple[int, int]]) -> bool:
    return any(start <= position < end for start, end in spans)


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Splits Markdown by heading and returns (heading, section).

    Ignora `#` dentro de blocos cercados por ```` ``` ```` (ex.: um `#` num
    A folder tree inside a ```text block is not a real heading.
    """
    fence_spans = _code_fence_spans(text)
    matches = [
        match
        for match in _MARKDOWN_HEADING_RE.finditer(text)
        if not _is_inside_any_span(match.start(), fence_spans)
    ]
    if not matches:
        return [("", text)] if text.strip() else []

    sections: list[tuple[str, str]] = []
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append(("", preamble))

    for index, match in enumerate(matches):
        heading = match.group(2).strip()
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((heading, text[start:end].strip()))
    return sections


_SQL_LEADING_COMMENTS_RE = re.compile(r"\A(?:[ \t]*--[^\n]*\n)+")
_SQL_CTE_BOUNDARY_RE = re.compile(r"\n(?=\s*\w+\s+as\s*\()", re.IGNORECASE)


def split_sql_blocks(text: str) -> list[tuple[str, str]]:
    """Splits SQL into CTE blocks while preserving header comments."""
    header_match = _SQL_LEADING_COMMENTS_RE.match(text)
    heading = header_match.group(0).strip() if header_match else ""
    body = text[header_match.end() :] if header_match else text

    blocks = [block for block in _SQL_CTE_BOUNDARY_RE.split(body) if block.strip()]
    if len(blocks) <= 1:
        return [(heading, text.strip())] if text.strip() else []
    return [(heading, block.strip()) for block in blocks]


def split_yaml_sections(text: str) -> list[tuple[str, str]]:
    """Splits YAML by top-level sections (unindented keys)."""
    sections: list[tuple[str, str]] = []
    current_heading = ""
    current_lines: list[str] = []

    def flush() -> None:
        block = "".join(current_lines).strip()
        if block:
            sections.append((current_heading, block))

    for line in text.splitlines(keepends=True):
        stripped = line.rstrip("\n")
        is_top_level = (
            bool(stripped)
            and not stripped[0].isspace()
            and not stripped.startswith("#")
        )
        if is_top_level:
            flush()
            current_lines = [line]
            current_heading = stripped.split(":")[0].strip()
        else:
            current_lines.append(line)
    flush()
    return sections if sections else ([("", text.strip())] if text.strip() else [])


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class DocumentChunk:
    """One row in the chunk table (design section 9.3 minimum contract)."""

    chunk_id: str
    source_path: str
    source_type: str
    source_layer: str
    object_name: str
    heading: str
    chunk_text: str
    content_hash: str
    indexed_at: datetime


def chunk_file(path: Path, repo_root: Path = REPO_ROOT) -> list[DocumentChunk]:
    """Reads a source file and returns its indexable chunks."""
    relative_path = path.relative_to(repo_root)
    suffix = relative_path.suffix.lower()
    text = path.read_text(encoding="utf-8")
    source_type, source_layer = classify_source(relative_path)
    object_name = object_name_for(relative_path)

    if suffix == ".md":
        sections = split_markdown_sections(text)
    elif suffix == ".sql":
        sections = split_sql_blocks(text)
    elif suffix in (".yml", ".yaml"):
        sections = split_yaml_sections(text)
    else:
        sections = [("", text)] if text.strip() else []

    indexed_at = datetime.now(timezone.utc).replace(tzinfo=None)
    chunks: list[DocumentChunk] = []
    chunk_index = 0
    for heading, section_text in sections:
        for piece in _split_by_size(section_text):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{relative_path.as_posix()}::chunk_{chunk_index:04d}",
                    source_path=relative_path.as_posix(),
                    source_type=source_type,
                    source_layer=source_layer,
                    object_name=object_name,
                    heading=heading,
                    chunk_text=piece,
                    content_hash=_hash(piece),
                    indexed_at=indexed_at,
                )
            )
            chunk_index += 1
    return chunks


def build_document_chunks(repo_root: Path = REPO_ROOT) -> list[DocumentChunk]:
    """Discovers source files and returns all repository chunks."""
    chunks: list[DocumentChunk] = []
    for path in discover_source_files(repo_root):
        chunks.extend(chunk_file(path, repo_root))
    return chunks


def _current_commit_sha(repo_root: Path = REPO_ROOT) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return result.stdout.strip()
    except Exception:
        return None


def persist_chunks(
    chunks: list[DocumentChunk], table: str = CHUNKS_TABLE
) -> dict[str, int | str]:
    """Persists chunks in a Delta table using MERGE (grain: `chunk_id`).

    A `chunk_id` no longer present in the current scan is removed, so the
    table always reflects the latest repository state rather than history.

    Unlike the notebooks in `ml/demand_forecasting`, this module also runs
    as a plain script (`spark_python_task`), where `spark` is not injected
    automatically as it is in a notebook. The session is therefore obtained
    explicitly here (`getOrCreate()` returns the active session in notebooks).
    """
    if not chunks:
        raise ValueError("No chunks were generated; persistence was aborted.")

    from pyspark.sql import SparkSession
    from pyspark.sql.types import (
        StringType,
        StructField,
        StructType,
        TimestampType,
    )

    spark = SparkSession.builder.getOrCreate()
    commit_sha = _current_commit_sha()
    rows = [
        {
            "chunk_id": chunk.chunk_id,
            "source_path": chunk.source_path,
            "source_type": chunk.source_type,
            "source_layer": chunk.source_layer,
            "object_name": chunk.object_name,
            "heading": chunk.heading,
            "chunk_text": chunk.chunk_text,
            "content_hash": chunk.content_hash,
            "commit_sha": commit_sha,
            "indexed_at": chunk.indexed_at,
        }
        for chunk in chunks
    ]
    chunks_schema = StructType(
        [
            StructField("chunk_id", StringType(), nullable=False),
            StructField("source_path", StringType(), nullable=False),
            StructField("source_type", StringType(), nullable=False),
            StructField("source_layer", StringType(), nullable=False),
            StructField("object_name", StringType(), nullable=False),
            StructField("heading", StringType(), nullable=False),
            StructField("chunk_text", StringType(), nullable=False),
            StructField("content_hash", StringType(), nullable=False),
            StructField("commit_sha", StringType(), nullable=True),
            StructField("indexed_at", TimestampType(), nullable=False),
        ]
    )
    chunks_df = spark.createDataFrame(rows, schema=chunks_schema)

    table_exists = spark.catalog.tableExists(table)
    if table_exists:
        from delta.tables import DeltaTable

        # CDF must be enabled before the MERGE so the vector index can see
        # this run's changes when it is synchronized later.
        _enable_change_data_feed(spark, table)
        target = DeltaTable.forName(spark, table)
        (
            target.alias("target")
            .merge(chunks_df.alias("source"), "target.chunk_id = source.chunk_id")
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .whenNotMatchedBySourceDelete()
            .execute()
        )
    else:
        chunks_df.write.format("delta").mode("overwrite").saveAsTable(table)
        _enable_change_data_feed(spark, table)

    persisted_df = spark.table(table)
    persisted_count = persisted_df.count()
    unique_chunk_count = persisted_df.select("chunk_id").distinct().count()
    expected_count = len(chunks)
    if persisted_count != expected_count or unique_chunk_count != expected_count:
        raise RuntimeError(
            "Persistence validation failed: "
            f"expected={expected_count}, rows={persisted_count}, "
            f"unique_chunk_ids={unique_chunk_count}"
        )

    return {
        "table": table,
        "rows": persisted_count,
        "unique_chunk_ids": unique_chunk_count,
    }


def _enable_change_data_feed(spark, table: str) -> None:
    """Enables CDF before a write that must be synchronized."""
    spark.sql(
        f"ALTER TABLE {table} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')"
    )


def main() -> None:
    files = discover_source_files()
    chunks = build_document_chunks()
    print(f"Files indexed: {len(files)}")
    print(f"Chunks generated: {len(chunks)}")
    summary = persist_chunks(chunks)
    print(f"Table: {summary['table']}")
    print(f"Rows persisted: {summary['rows']}")
    print(f"Unique chunk IDs: {summary['unique_chunk_ids']}")
    print("MARINA INDEXING COMPLETE")


if __name__ == "__main__":
    main()
