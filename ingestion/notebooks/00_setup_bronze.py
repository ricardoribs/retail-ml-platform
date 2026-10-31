# Databricks notebook source

# COMMAND ----------

"""Prepares Bronze ingestion dependencies for Databricks.

Loads table configuration, builds JDBC connection settings, ensures that the Bronze
ingestion control table exists, and validates SQL Server connectivity. Run this
notebook before Bronze ingestion jobs."""

import json  # noqa: I001
import os
import sys

from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    dbutils: Any
    spark: Any

PROJECT_ROOT = Path(os.path.abspath(".."))
sys.path.insert(0, str(PROJECT_ROOT))

from src.connection import get_jdbc_config, test_connection  # noqa: E402
from src.control import ensure_control_table  # noqa: E402

# COMMAND ----------

# Bundle parameter
dbutils.widgets.text("config_path", "../config/tables_config.json")  # noqa: F821
CONFIG_PATH = dbutils.widgets.get("config_path")  # noqa: F821
SECRET_SCOPE = "retail-analytics-source"
CONTROL_TABLE = "retail_analytics.bronze.ingestion_control"

# COMMAND ----------

config_path = Path(CONFIG_PATH)

assert config_path.exists(), (
    f"[ERROR] tables_config.json not found at: {CONFIG_PATH}"
)

with open(config_path, "r", encoding="utf-8") as f:
    tables_config = json.load(f)

tables = tables_config.get("tables", {})
print(f"[OK] Loaded tables_config.json: {len(tables)} tables")

# COMMAND ----------

jdbc_url, jdbc_props = get_jdbc_config(
    dbutils,  # noqa: F821
    secret_scope=SECRET_SCOPE,
)

# Print only host:port to avoid exposing credentials in the output
print(f"[OK] JDBC: {jdbc_url.split(';')[0]}")

# COMMAND ----------

# Create the catalog, schema, and control table if they do not exist
ensure_control_table(spark, control_table=CONTROL_TABLE)  # noqa: F821

print(f"[OK] Control table: {CONTROL_TABLE}")
spark.sql(f"DESCRIBE {CONTROL_TABLE}").show(truncate=False)  # noqa: F821

# COMMAND ----------

# Check that the cluster can reach SQL Server
ok = test_connection(spark, dbutils, secret_scope=SECRET_SCOPE)  # noqa: F821

assert ok, "[ERROR] Connection failed. Check the Secret Scope, host, and credentials."

# COMMAND ----------

print("BRONZE SETUP COMPLETE")
print(f"  Config   : {CONFIG_PATH}")
print(f"  Tables   : {len(tables)}")
print(f"  Control  : {CONTROL_TABLE}")
print("  Connection: OK")
