"""Sanity checks for src/config.py.

Outside Databricks, dbutils is unavailable: the NameError fallback must
always resolve to the defaults documented in the README.

The import used here is the same one used by notebooks and scripts
(`ml.demand_forecasting.src.config`), so these tests also guard the import
mechanism against regressions.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from ml.demand_forecasting.src import config as forecast_config
from ml.demand_forecasting.src.config import (
    CATALOG,
    FORECAST_CONFIG,
    MARTS_SCHEMA,
    build_forecast_config,
    fq_table,
    resolve_reference_date,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_default_catalog_and_schema_match_readme():
    assert CATALOG == "retail_analytics"
    assert MARTS_SCHEMA == "default_marts"


def test_fq_table_builds_fully_qualified_name():
    assert fq_table(MARTS_SCHEMA, "fct_sales") == "retail_analytics.default_marts.fct_sales"


def test_reference_date_builds_august_snapshot_and_next_three_months():
    config = build_forecast_config("2026-09-13", execution_id="run-1")

    assert config.reference_date == date(2026, 9, 13)
    assert config.cutoff_date == date(2026, 8, 31)
    assert config.horizon_months == ("2026-09", "2026-10", "2026-11")
    assert config.snapshot_id == "forecast_2026_08"
    assert config.execution_id == "run-1"


def test_reference_date_builds_september_snapshot_and_next_three_months():
    config = build_forecast_config("2026-10-13")

    assert config.cutoff_date == date(2026, 9, 30)
    assert config.horizon_months == ("2026-10", "2026-11", "2026-12")
    assert config.snapshot_id == "forecast_2026_09"


def test_same_cutoff_has_stable_snapshot_across_retries():
    first = build_forecast_config("2026-09-01", execution_id="attempt-1")
    retry = build_forecast_config("2026-09-30", execution_id="attempt-2")

    assert first.snapshot_id == retry.snapshot_id == "forecast_2026_08"
    assert first.execution_id != retry.execution_id


def test_blank_reference_date_uses_sao_paulo_operational_date():
    instant = datetime(2026, 9, 14, 2, 30, tzinfo=timezone.utc)

    assert resolve_reference_date("", now=instant) == date(2026, 9, 13)


def test_forecast_config_keeps_business_rules():
    assert FORECAST_CONFIG.order_status_filter == "paid"


def test_regression_eligibility_thresholds_match_dbt_model():
    # Same rule as int_monthly_product_location_demand.sql (is_regression_eligible):
    # history_month_count >= 6 and positive_demand_month_count >= 3.
    assert FORECAST_CONFIG.min_history_months == 6
    assert FORECAST_CONFIG.min_positive_demand_months == 3


def test_import_works_outside_pytest():
    """Ensure the module imports without pytest installed.


    Simulate a Databricks notebook or script executed from the repository root,
    where the repository root is the only available path in sys.path.
    Passing this test confirms that the README notebook instructions work.
    """
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from ml.demand_forecasting.src.config import build_forecast_config;"
            "print(build_forecast_config('2026-09-13').cutoff_date)",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "2026-08-31" in result.stdout


def test_set_task_value_uses_resolved_dbutils(monkeypatch):
    calls = []

    class TaskValues:
        def set(self, **kwargs):
            calls.append(kwargs)

    class Utils:
        class jobs:
            taskValues = TaskValues()

    monkeypatch.setattr(forecast_config, "_databricks_utils", lambda: Utils())

    assert forecast_config.set_task_value("release_status", "false", required=True)
    assert calls == [{"key": "release_status", "value": "false"}]


def test_required_task_value_fails_closed_when_dbutils_is_unavailable(monkeypatch):
    monkeypatch.setattr(forecast_config, "_databricks_utils", lambda: None)

    assert forecast_config.set_task_value("optional", "value") is False
    with pytest.raises(RuntimeError, match="DBUtils is unavailable"):
        forecast_config.set_task_value("required", "value", required=True)
