"""Bronze ingestion control table.

Stores watermarks and per-table status, supporting idempotency and recovery
after failures."""

from datetime import datetime, timezone
from typing import Optional


def ensure_control_table(spark, control_table: str) -> None:
    """Ensures the catalog, schema, and control table exist before ingestion."""
    catalog, schema, table = control_table.split(".")

    spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")

    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")

    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {control_table} (
            table_name           STRING NOT NULL,
            last_watermark_value STRING,      -- Latest value read (e.g. max updated_at)
            last_run_at          TIMESTAMP,
            last_status          STRING,
            rows_processed       BIGINT,
            error_message        STRING
        ) USING DELTA
    """)


def get_last_watermark(spark, control_table: str, table_name: str) -> Optional[str]:
    """Returns the last saved watermark, or `None` for the first (full) load."""
    row = spark.sql(f"""
        SELECT last_watermark_value
        FROM {control_table}
        WHERE table_name = '{table_name}'
    """).collect()

    if not row:
        return None
    return row[0]["last_watermark_value"]


def update_control(
    spark,
    control_table: str,
    table_name: str,
    new_watermark: Optional[str],
    status: str,
    rows_processed: int = 0,
    error_message: Optional[str] = None,
) -> None:
    """Records a table's ingestion result (watermark, status, processed rows,
    and error) with an idempotent upsert to the control table."""
    now = datetime.now(timezone.utc).isoformat()

    # Escape single quotes to avoid breaking the inline SQL.
    safe_err = (error_message or "").replace("'", "''")
    safe_wm = (new_watermark or "").replace("'", "''")

    # Idempotent upsert: update an existing row or insert the first one.
    spark.sql(f"""
        MERGE INTO {control_table} AS target
        USING (
            SELECT
                '{table_name}'                AS table_name,
                {
        ("NULL" if new_watermark is None else f"'{safe_wm}'")
    } AS last_watermark_value,
                CAST('{now}' AS TIMESTAMP)    AS last_run_at,
                '{status}'                    AS last_status,
                CAST({rows_processed} AS BIGINT) AS rows_processed,
                {
        ("NULL" if error_message is None else f"'{safe_err}'")
    } AS error_message
        ) AS source
        ON target.table_name = source.table_name
        WHEN MATCHED THEN UPDATE SET
            last_watermark_value = source.last_watermark_value,
            last_run_at          = source.last_run_at,
            last_status          = source.last_status,
            rows_processed       = source.rows_processed,
            error_message        = source.error_message
        WHEN NOT MATCHED THEN INSERT *
    """)
