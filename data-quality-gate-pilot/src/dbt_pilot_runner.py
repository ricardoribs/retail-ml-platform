# Databricks notebook source
# MAGIC %md
# MAGIC # Data Quality Gate dbt pilot runner
# MAGIC
# MAGIC This notebook runs only the controlled dbt test fixture used by the
# MAGIC Data Quality Gate V2 pilot. It never runs ingestion, forecasting, or
# MAGIC deployment tasks.

# COMMAND ----------

import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

DEFAULT_CATALOG = "retail_analytics"
DEFAULT_SCHEMA = "data_quality_gate_pilot"
DEFAULT_SELECTOR = "path:tests/pilot"
PILOT_SCHEMA_PREFIX = "data_quality_gate_pilot_"
SECRET_SCOPE = "retail_analytics"
_DBT_DIAGNOSTIC_PATTERN = re.compile(
    r"(?:database error|compilation error|runtime error|failure in test|"
    r"encountered an error|\bFAIL\s+\d+\b|completed with \d+ error)",
    re.IGNORECASE,
)
_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_SENSITIVE_VALUE_PATTERN = re.compile(
    r"(?i)\b(token|password|secret)\s*([=:])\s*[^\s,;]+"
)


def widget_value(name: str, default: str) -> str:
    """Read a notebook widget while preserving a safe default for manual runs."""
    dbutils.widgets.text(name, default)  # noqa: F821
    return dbutils.widgets.get(name).strip()  # noqa: F821


def boolean_widget_value(name: str, default: str) -> bool:
    """Read a strict boolean widget value for the controlled test mode."""
    value = widget_value(name, default).lower()
    if value not in {"true", "false"}:
        raise ValueError(
            f"Widget {name!r} must be 'true' or 'false'; received {value!r}."
        )
    return value == "true"


def require_pilot_destination(catalog: str, schema: str) -> None:
    """Prevent this runner from being aimed at the historic default schema."""
    if catalog != DEFAULT_CATALOG:
        raise ValueError(
            f"Pilot catalog must be {DEFAULT_CATALOG!r}; received {catalog!r}."
        )
    if not schema.startswith(PILOT_SCHEMA_PREFIX):
        raise ValueError(
            "Pilot schema must begin with "
            f"{PILOT_SCHEMA_PREFIX!r}; received {schema!r}."
        )


def find_dbt_project_dir() -> Path:
    """Find the synchronized dbt project beside the standalone pilot bundle."""
    notebook_path = (
        Path("/Workspace")
        / dbutils.notebook.entry_point.getDbutils()  # noqa: F821
        .notebook()
        .getContext()
        .notebookPath()
        .get()
        .lstrip("/")
    )

    for directory in (notebook_path.parent, *notebook_path.parents):
        project_dir = directory / "dbt"
        if (project_dir / "dbt_project.yml").is_file():
            return project_dir

    raise RuntimeError("Could not locate the synchronized dbt project.")


def dbt_diagnostic(stdout: str, stderr: str) -> str:
    """Return a narrow, safe diagnostic instead of emitting command output verbatim."""
    lines = (stdout + "\n" + stderr).splitlines()
    for index, line in enumerate(lines):
        normalized_line = _ANSI_ESCAPE_PATTERN.sub("", line).strip()
        if _DBT_DIAGNOSTIC_PATTERN.search(normalized_line):
            diagnostic_lines = [normalized_line]
            for detail_line in lines[index + 1 : index + 6]:
                normalized_detail = _ANSI_ESCAPE_PATTERN.sub("", detail_line).strip()
                if normalized_detail:
                    diagnostic_lines.append(normalized_detail)
            diagnostic = " | ".join(diagnostic_lines)
            return _SENSITIVE_VALUE_PATTERN.sub(r"\1\2[REDACTED]", diagnostic)[:1000]
    return (
        "No allowlisted dbt diagnostic was available. "
        "Inspect the task run in Databricks."
    )


def run(
    command: list[str], *, cwd: Path, environment: dict[str, str] | None = None
) -> None:
    """Run a command and retain the safe dbt diagnostic in the Job run log."""
    printable = " ".join(command)
    print(f"\n{'=' * 72}\nRunning: {printable}\n{'=' * 72}")
    result = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        diagnostic = dbt_diagnostic(result.stdout, result.stderr)
        raise RuntimeError(
            f"Command failed with exit code {result.returncode}: {printable}\n"
            f"Safe dbt diagnostic: {diagnostic}"
        )

    print(result.stdout or "(no stdout)")
    if result.stderr:
        print(result.stderr, file=sys.stderr)


def connection_profile(*, catalog: str, schema: str) -> str:
    """Build an ephemeral dbt profile without logging or persisting secrets."""
    host = dbutils.secrets.get(scope=SECRET_SCOPE, key="databricks_host")  # noqa: F821
    http_path = dbutils.secrets.get(  # noqa: F821
        scope=SECRET_SCOPE,
        key="databricks_http_path",
    )
    token = dbutils.secrets.get(scope=SECRET_SCOPE, key="databricks_token")  # noqa: F821

    return textwrap.dedent(
        f"""
        retail_analytics:
          target: data_quality_gate_pilot
          outputs:
            data_quality_gate_pilot:
              type: databricks
              catalog: {json.dumps(catalog)}
              schema: {json.dumps(schema)}
              host: {json.dumps(host)}
              http_path: {json.dumps(http_path)}
              token: {json.dumps(token)}
              threads: 1
        """
    ).lstrip()


catalog = widget_value("dbt_catalog", DEFAULT_CATALOG)
schema = widget_value("dbt_schema", DEFAULT_SCHEMA)
selector = widget_value("dbt_select", DEFAULT_SELECTOR)
inject_failure = boolean_widget_value("inject_failure", "false")
if selector != DEFAULT_SELECTOR:
    raise ValueError(
        f"Pilot selector must be {DEFAULT_SELECTOR!r}; received {selector!r}."
    )
require_pilot_destination(catalog, schema)

project_dir = find_dbt_project_dir()
print(f"dbt project: {project_dir}")
print(f"dbt catalog: {catalog}")
print(f"dbt schema: {schema}")
print(f"dbt selector: {selector}")
print(f"controlled failure: {inject_failure}")

run(
    [sys.executable, "-m", "pip", "install", "dbt-databricks", "--quiet"],
    cwd=project_dir,
)

with tempfile.TemporaryDirectory(
    prefix="data-quality-gate-dbt-profiles-"
) as profiles_dir:
    profiles_path = Path(profiles_dir) / "profiles.yml"
    profiles_path.write_text(
        connection_profile(catalog=catalog, schema=schema), encoding="utf-8"
    )

    dbt_environment = os.environ.copy()
    dbt_environment["DBT_PROFILES_DIR"] = profiles_dir
    run(
        [sys.executable, "-m", "dbt.cli.main", "deps"],
        cwd=project_dir,
        environment=dbt_environment,
    )
    test_command = [sys.executable, "-m", "dbt.cli.main", "test", "--select", selector]
    if inject_failure:
        test_command.extend(["--vars", "data_quality_gate_inject_failure: true"])
    run(test_command, cwd=project_dir, environment=dbt_environment)

print("Data Quality Gate pilot dbt validation completed successfully.")
