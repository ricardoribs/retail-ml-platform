# Databricks notebook source
#  %md
#  01 - Bronze ingestion (SQL Server -> Delta)
#  Extracts the tables configured in `config/tables_config.json` from the
#  `retail_analytics` SQL Server database and loads them into the Lakehouse
#  Bronze layer as Delta tables. Loads are incremental by watermark and
#  idempotent: each table is processed independently, and progress is saved
#  to `ingestion_control` only after success.
#  **Widgets**
#   `table_name`: a specific table to reprocess/debug, or `all` (default) to
#   run the complete pipeline.
#  `config_path`: path to the table configuration file.
#  `config_path`: path to the table configuration file.

# COMMAND ----------

"""Runs Bronze ingestion from SQL Server into Delta tables.

Loads the table configuration selected through Databricks widgets, runs the Bronze
ingestion pipeline, displays each table's results, and raises an error if any
configured table fails."""
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    dbutils: Any
    display: Any
    spark: Any

# Databricks global variables (injected by the environment)
dbutils.library.restartPython()  # noqa: F821

# COMMAND ----------

import json  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402

sys.path.append(os.path.abspath(".."))

from src.ingestion import run_ingestion  # noqa: E402

# COMMAND ----------

dbutils.widgets.text("config_path", "../config/tables_config.json")  # noqa: F821
dbutils.widgets.text("table_name", "all")  # noqa: F821

config_path = dbutils.widgets.get("config_path")  # noqa: F821
table_name = dbutils.widgets.get("table_name")  # noqa: F821

with open(config_path) as f:
    config = json.load(f)

# COMMAND ----------

results = run_ingestion(
    spark,  # noqa: F821
    dbutils,  # noqa: F821
    config,
    table_filter=table_name,
)

# COMMAND ----------

from pyspark.sql.types import (  # noqa: E402, I001
    LongType,
    StringType,
    StructField,
    StructType,
)

results_schema = StructType(
    [
        StructField("table", StringType(), True),
        StructField("status", StringType(), True),
        StructField("rows", LongType(), True),
        StructField("error", StringType(), True),
    ]
)
results_df = spark.createDataFrame(results, schema=results_schema)  # noqa: F821
display(results_df)  # noqa: F821

# COMMAND ----------

failed = [r for r in results if r["status"] == "failed"]
total_rows = sum(r["rows"] for r in results if r["status"] == "success")

print(
    f"Tables processed: {len(results)} | "
    f"succeeded: {len(results) - len(failed)} | "
    f"failed: {len(failed)}"
)
print(f"Total rows loaded/updated in this run: {total_rows}")

if failed:
    raise RuntimeError(f"Table ingestion failed: {[r['table'] for r in failed]}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM retail_analytics.bronze.ingestion_control ORDER BY table_name;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT COUNT(*) FROM retail_analytics.bronze.suppliers;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM retail_analytics.bronze.products LIMIT 10;
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC SHOW TABLES IN retail_analytics.bronze;
# MAGIC
