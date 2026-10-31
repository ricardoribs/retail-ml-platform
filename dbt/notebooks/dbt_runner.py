# Databricks notebook source
# MAGIC %md
# MAGIC # dbt Runner - Run Transformations
# MAGIC
# MAGIC This notebook runs the complete dbt pipeline:
# MAGIC 1. Installs dbt-databricks
# MAGIC 2. Finds the dbt project directory
# MAGIC 3. Reads credentials from a Secret Scope and generates profiles.yml
# MAGIC 4. Installs dependencies (dbt deps)
# MAGIC 5. Compiles models (dbt compile)
# MAGIC 6. Runs models (dbt run)
# MAGIC 7. Runs tests (dbt test)

# COMMAND ----------

# DBTITLE 1,Install dbt-databricks
import json
import os
import subprocess
import sys
import textwrap

DBT_VARIABLE_ARGS = []
DBT_SELECT = ""

print("Installing dbt-databricks...")
install_result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "dbt-databricks", "--quiet"],
    capture_output=True,
    text=True,
)
print(install_result.stdout)
if install_result.returncode != 0:
    print(install_result.stderr)
    raise Exception("Failed to install dbt-databricks")
print("dbt-databricks installed successfully")

# COMMAND ----------


# DBTITLE 1,Helper functions
def find_dbt_project_dir():
    """
    Walks up directories from the notebook's actual path (via the API).

    With an Asset Bundle, this path points to a writable area in the
    workspace.
    """
    notebook_path = (
        dbutils.notebook.entry_point.getDbutils()  # noqa: F821
        .notebook()
        .getContext()
        .notebookPath()
        .get()
    )
    workspace_root = "/Workspace" + notebook_path
    current = os.path.dirname(workspace_root)

    while current != "/":
        candidate = os.path.join(current, "dbt_project.yml")
        if os.path.exists(candidate):
            return current
        current = os.path.dirname(current)

    raise Exception(
        f"dbt_project.yml not found while walking up from {workspace_root}"
    )


def run_dbt_command(command_args):
    """
    Runs a dbt command through 'python -m dbt.cli.main', ensuring it uses
    the same Python interpreter where dbt-databricks was installed.
    """
    effective_args = list(command_args)
    if effective_args and effective_args[0] in {"compile", "run", "test", "parse"}:
        effective_args.extend(DBT_VARIABLE_ARGS)
        if DBT_SELECT:
            effective_args.extend(["--select", DBT_SELECT])
    full_command = [sys.executable, "-m", "dbt.cli.main"] + effective_args
    print(f"\n{'=' * 70}")
    print(f"Running: {' '.join(full_command)}")
    print(f"{'=' * 70}\n")

    result = subprocess.run(
        full_command, capture_output=True, text=True, cwd=DBT_PROJECT_DIR
    )

    print("STDOUT:")
    print(result.stdout if result.stdout else "(empty)")

    print("\nSTDERR:")
    print(result.stderr if result.stderr else "(empty)")

    if result.returncode != 0:
        print(f"\nERROR: Command failed with exit code {result.returncode}")
        diagnostic = "\n".join(
            part for part in (result.stdout.strip(), result.stderr.strip()) if part
        )
        raise RuntimeError(
            f"dbt command failed ({result.returncode}): {' '.join(full_command)}\n"
            f"dbt diagnostics (last 8,000 characters):\n{diagnostic[-8000:]}"
        )

    print("\nCommand completed successfully!")
    return result.returncode


DBT_PROJECT_DIR = find_dbt_project_dir()
print(f"dbt project found at: {DBT_PROJECT_DIR}")

REPO_ROOT = os.path.dirname(DBT_PROJECT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from ml.demand_forecasting.src.config import (  # noqa: E402
    FORECAST_CONFIG,
    set_task_value,
)

DBT_VARIABLE_ARGS = [
    "--vars",
    json.dumps(
        {
            "forecast_config": {
                "reference_date_sql": (
                    f"date '{FORECAST_CONFIG.reference_date.isoformat()}'"
                )
            }
        }
    ),
]
try:
    DBT_SELECT = dbutils.widgets.get("dbt_select")  # noqa: F821
except Exception:
    dbutils.widgets.text("dbt_select", "")  # noqa: F821
    DBT_SELECT = dbutils.widgets.get("dbt_select")  # noqa: F821
DBT_SELECT = DBT_SELECT.strip()

print("Forecast time context")
print(f"  Reference date: {FORECAST_CONFIG.reference_date}")
print(f"  Cutoff: {FORECAST_CONFIG.cutoff_date}")
print(f"  Horizons: {', '.join(FORECAST_CONFIG.horizon_months)}")
print(f"  Snapshot ID: {FORECAST_CONFIG.snapshot_id}")
print(f"  Execution ID: {FORECAST_CONFIG.execution_id}")
if DBT_SELECT:
    print(f"  dbt selector: {DBT_SELECT}")

set_task_value("reference_date", FORECAST_CONFIG.reference_date.isoformat())
set_task_value("cutoff_date", FORECAST_CONFIG.cutoff_date.isoformat())
set_task_value("snapshot_id", FORECAST_CONFIG.snapshot_id)
set_task_value("execution_id", FORECAST_CONFIG.execution_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read credentials from the Secret Scope and generate profiles.yml

# COMMAND ----------

SECRET_SCOPE = "retail_analytics"

try:
    databricks_host = dbutils.secrets.get(  # noqa: F821
        scope=SECRET_SCOPE, key="databricks_host"
    )
    databricks_http_path = dbutils.secrets.get(  # noqa: F821
        scope=SECRET_SCOPE, key="databricks_http_path"
    )
    databricks_token = dbutils.secrets.get(  # noqa: F821
        scope=SECRET_SCOPE, key="databricks_token"
    )
    print(f"Credentials read successfully from Secret Scope '{SECRET_SCOPE}'!")
except Exception as e:
    print(f"ERROR: Could not read secrets from scope '{SECRET_SCOPE}'")
    print("\nHow to configure:")
    print(
        "1. Create the scope if it does not exist: "
        "#secrets/createScope in the workspace URL"
    )
    print("2. Add secrets through the Databricks SDK or CLI:")
    print(f"   databricks secrets put-secret {SECRET_SCOPE} databricks_host")
    print(f"   databricks secrets put-secret {SECRET_SCOPE} databricks_http_path")
    print(f"   databricks secrets put-secret {SECRET_SCOPE} databricks_token")
    raise Exception(f"Failed to read secrets: {e}")

print(f"  HOST: {databricks_host}")
print(f"  HTTP_PATH: {databricks_http_path}")
print("  TOKEN: ***")

profiles_dir = "/tmp/dbt_profiles"
os.makedirs(profiles_dir, exist_ok=True)

profiles_content = textwrap.dedent(f"""
retail_analytics:
  target: prod
  outputs:
    prod:
      type: databricks
      catalog: retail_analytics
      schema: default
      host: "{databricks_host}"
      http_path: "{databricks_http_path}"
      token: "{databricks_token}"
      threads: 4
""")

with open(f"{profiles_dir}/profiles.yml", "w") as f:
    f.write(profiles_content)

os.environ["DBT_PROFILES_DIR"] = profiles_dir
print(f"\nprofiles.yml generated successfully at: {profiles_dir}")

print("\nprofiles.yml contents (token redacted):")
print(profiles_content.replace(databricks_token, "***"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install dependencies

# COMMAND ----------

print("Installing dbt packages...")
run_dbt_command(["deps"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compile models (validate SQL)

# COMMAND ----------

print("Compiling models and validating SQL...")
run_dbt_command(["compile"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run models (create tables/views)

# COMMAND ----------

print("Running dbt models...")
run_dbt_command(["run"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run tests

# COMMAND ----------

print("Running dbt tests...")
run_dbt_command(["test"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pipeline complete

# COMMAND ----------

print("\n" + "=" * 70)
print("DBT PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\nResumo:")
print("  - Dependencies installed")
print("  - Models compiled")
print("  - Models executed")
print("  - Tests passed")
print("\n" + "=" * 70)
