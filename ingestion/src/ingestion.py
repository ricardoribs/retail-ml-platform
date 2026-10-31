"""Generic logic for ingesting data into the Bronze layer.

Each table in `tables_config.json` is processed by `ingest_table`, which:
1. reads the latest watermark from the control table;
2. extracts new or changed data from SQL Server via JDBC;
3. writes Delta data to Bronze (overwrite, MERGE upsert, or insert-only);
4. updates the control table with the new watermark only after success."""

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from .connection import get_jdbc_config
from .control import (
    ensure_control_table,
    get_last_watermark,
    update_control,
)

SOURCE_SYSTEM = "sqlserver_retail"


def _build_filter_sql(
    source_schema: str,
    table_cfg: Dict[str, Any],
    last_watermark: Optional[str],
) -> str:
    """Builds the source-table extraction SQL with an incremental watermark
    filter, or no filter for a `full` load or the first run."""
    table_name = table_cfg["name"]
    load_type = table_cfg["load_type"]
    watermark_column = table_cfg.get("watermark_column")

    base = f"SELECT * FROM {source_schema}.{table_name}"

    if load_type == "full" or last_watermark is None:
        return base

    # Escape single quotes to prevent SQL errors.
    if watermark_column == "id":
        # Do not quote numeric IDs.
        safe_watermark = str(last_watermark).replace("'", "''")
        return f"{base} WHERE id > {safe_watermark}"

    # Quote and escape timestamps/dates.
    safe_watermark = str(last_watermark).replace("'", "''")
    return f"{base} WHERE {watermark_column} > '{safe_watermark}'"


def _get_partition_bounds(
    spark: SparkSession,
    jdbc_url: str,
    jdbc_props: Dict[str, Any],
    filter_sql: str,
    column: str,
) -> Tuple[Optional[Any], Optional[Any]]:
    """Calculates the partition column bounds (min/max) within the filtered
    data, for the parallel JDBC read's `lowerBound`/`upperBound`."""
    bounds_sql = (
        f"(SELECT MIN({column}) AS lo, MAX({column}) AS hi "
        f"FROM ({filter_sql}) AS t) AS bounds"
    )
    row = (
        spark.read.format("jdbc")
        .option("url", jdbc_url)
        .options(**jdbc_props)
        .option("dbtable", bounds_sql)
        .load()
        .collect()[0]
    )
    return row["lo"], row["hi"]


def read_source_table(
    spark: SparkSession,
    jdbc_url: str,
    jdbc_props: Dict[str, Any],
    source_schema: str,
    table_cfg: Dict[str, Any],
    last_watermark: Optional[str],
) -> DataFrame:
    """Reads the source table over JDBC with an incremental filter and,
    when configured, partitioned reads for larger tables."""
    filter_sql = _build_filter_sql(source_schema, table_cfg, last_watermark)

    reader = spark.read.format("jdbc").option("url", jdbc_url).options(**jdbc_props)

    partition_cfg = table_cfg.get("partition_read")
    if partition_cfg:
        column = partition_cfg["column"]
        lower, upper = _get_partition_bounds(
            spark, jdbc_url, jdbc_props, filter_sql, column
        )
        if lower is not None and upper is not None:
            reader = (
                reader.option("partitionColumn", column)
                .option("lowerBound", str(lower))
                .option("upperBound", str(int(upper) + 1))
                .option("numPartitions", str(partition_cfg["num_partitions"]))
            )

    return reader.option("dbtable", f"({filter_sql}) AS src").load()


def add_technical_columns(df: DataFrame) -> DataFrame:
    """Adds the technical lineage columns `_ingested_at` and `_source_system`
    to the extracted DataFrame."""
    return df.withColumn("_ingested_at", F.current_timestamp()).withColumn(
        "_source_system", F.lit(SOURCE_SYSTEM)
    )


def write_full(df: DataFrame, catalog: str, schema: str, table_name: str) -> None:
    """Overwrites the Bronze Delta table with the complete DataFrame contents."""
    target = f"{catalog}.{schema}.{table_name}"
    df.write.format("delta").mode("overwrite").option(
        "overwriteSchema", "true"
    ).saveAsTable(target)


def write_merge(
    spark: SparkSession,
    df: DataFrame,
    catalog: str,
    schema: str,
    table_name: str,
    primary_key: List[str],
    insert_only: bool = False,
) -> None:
    """Writes the DataFrame to Bronze with a MERGE on the primary key: upsert
    (update + insert) by default, or insert-only when `insert_only=True`.
    On the first run, creates the table."""
    target = f"{catalog}.{schema}.{table_name}"
    view_name = f"_staging_{table_name}"
    df.createOrReplaceTempView(view_name)

    if not spark.catalog.tableExists(target):
        df.write.format("delta").option("mergeSchema", "true").saveAsTable(target)
        return

    on_clause = " AND ".join(f"tgt.{pk} = src.{pk}" for pk in primary_key)
    merge_sql = f"MERGE INTO {target} AS tgt USING {view_name} AS src ON {on_clause} "
    if not insert_only:
        merge_sql += "WHEN MATCHED THEN UPDATE SET * "
    merge_sql += "WHEN NOT MATCHED THEN INSERT *"

    spark.sql(merge_sql)


def compute_new_watermark(
    df: DataFrame, watermark_column: Optional[str]
) -> Optional[str]:
    """Calculates the new watermark as the maximum value in the extracted
    batch. Returns `None` if the table has no watermark or the batch is empty
    (in which case the caller keeps the previous watermark)."""
    if watermark_column is None:
        return None

    value = df.agg(F.max(F.col(watermark_column)).alias("max_wm")).collect()[0][
        "max_wm"
    ]
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S.%f")
    return str(value)


def ingest_table(
    spark: SparkSession,
    jdbc_url: str,
    jdbc_props: Dict[str, Any],
    source_schema: str,
    bronze_catalog: str,
    bronze_schema: str,
    control_table: str,
    table_cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Ingests one table and updates the control table.

    On failure, the previous watermark is preserved so the next run resumes
    from the same point (idempotency).
    """
    table_name = table_cfg["name"]
    load_type = table_cfg["load_type"]
    watermark_column = table_cfg.get("watermark_column")
    primary_key = table_cfg.get("primary_key", [])

    last_watermark = get_last_watermark(spark, control_table, table_name)

    try:
        df = add_technical_columns(
            read_source_table(
                spark,
                jdbc_url,
                jdbc_props,
                source_schema,
                table_cfg,
                last_watermark,
            )
        )

        rows = df.count()

        if load_type == "full":
            write_full(df, bronze_catalog, bronze_schema, table_name)
            new_watermark = None
        else:
            write_merge(
                spark,
                df,
                bronze_catalog,
                bronze_schema,
                table_name,
                primary_key,
                insert_only=(load_type == "merge_insert_only"),
            )
            new_watermark = (
                compute_new_watermark(df, watermark_column) or last_watermark
            )

        update_control(
            spark,
            control_table,
            table_name,
            new_watermark,
            "success",
            rows_processed=rows,
            error_message=None,
        )

        return {"table": table_name, "status": "success", "rows": rows, "error": None}

    except Exception as exc:  # noqa: BLE001
        # A table failure must not stop the pipeline.
        update_control(
            spark,
            control_table,
            table_name,
            last_watermark,
            "failed",
            rows_processed=0,
            error_message=str(exc),
        )
        return {"table": table_name, "status": "failed", "rows": 0, "error": str(exc)}


def run_ingestion(
    spark: SparkSession,
    dbutils: Any,
    config: Dict[str, Any],
    table_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Ingests all tables, or a single table when `table_filter` is set, and
    returns the result for each one."""
    general = config["general"]
    source_schema = general["source_schema"]
    bronze_catalog = general["bronze_catalog"]
    bronze_schema = general["bronze_schema"]
    control_table = general["control_table"]

    jdbc_url, jdbc_props = get_jdbc_config(dbutils, general["secret_scope"])

    ensure_control_table(spark, control_table)

    tables = config["tables"]
    if table_filter and table_filter != "all":
        tables = [t for t in tables if t["name"] == table_filter]

    results = []
    for table_cfg in tables:
        result = ingest_table(
            spark,
            jdbc_url,
            jdbc_props,
            source_schema,
            bronze_catalog,
            bronze_schema,
            control_table,
            table_cfg,
        )
        results.append(result)

    return results
