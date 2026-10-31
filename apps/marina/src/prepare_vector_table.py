"""Apply the metadata required by the Delta Sync index once per environment."""

from __future__ import annotations

import sys
from pathlib import Path


def _resolve_repo_root() -> Path:
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

SOURCE_TABLE = fq_table(MARTS_SCHEMA, "marina_document_chunks")


def prepare_table(spark, table: str = SOURCE_TABLE) -> None:
    """Enables CDF and declares `chunk_id` as the primary key."""
    catalog, schema, table_name = table.split(".", maxsplit=2)
    spark.sql(
        f"ALTER TABLE {table} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')"
    )

    columns = spark.catalog.listColumns(table)
    chunk_column = next(
        (column for column in columns if column.name == "chunk_id"), None
    )
    if chunk_column is None:
        raise RuntimeError(f"Table {table} does not contain the chunk_id column.")
    if chunk_column.nullable:
        spark.sql(f"ALTER TABLE {table} ALTER COLUMN chunk_id SET NOT NULL")

    constraints = spark.sql(
        f"""
        SELECT constraint_name
        FROM `{catalog}`.information_schema.table_constraints
        WHERE table_schema = '{schema}'
          AND table_name = '{table_name}'
          AND constraint_type = 'PRIMARY KEY'
        """
    ).collect()
    if not constraints:
        spark.sql(
            f"ALTER TABLE {table} ADD CONSTRAINT {table_name}_pk PRIMARY KEY (chunk_id)"
        )


def main() -> None:
    from pyspark.sql import SparkSession

    prepare_table(SparkSession.builder.getOrCreate())
    print(f"Table prepared for AI Search: {SOURCE_TABLE}")


if __name__ == "__main__":
    main()
