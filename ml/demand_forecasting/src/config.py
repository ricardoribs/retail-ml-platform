"""Central configuration for catalog, schema, and demand-forecast parameters.

The single source of truth for Unity Catalog fully qualified names and the
parameters documented in ../README.md (section 3): operational date,
dynamic cutoff, forecast horizon, and regression eligibility thresholds.
New scripts and notebooks should import these values instead of duplicating
them, keeping them aligned with the dbt model
`int_monthly_product_location_demand.sql` and the README.

Can be re-run in a Databricks notebook (reads `dbutils.widgets`, allowing a
different catalog or schema without code changes) or locally/in tests (falls
back to defaults via `try/except NameError`). The same pattern used in
`notebooks/00_data_audit.py` is centralized here to avoid duplication."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

OPERATIONAL_TIMEZONE = "America/Sao_Paulo"


def _databricks_utils():
    """Resolve DBUtils when this module is imported by a notebook as well."""
    try:
        return dbutils  # noqa: F821
    except NameError:
        pass

    try:
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession

        spark_session = SparkSession.getActiveSession()
        return DBUtils(spark_session) if spark_session is not None else None
    except Exception:
        return None


def _widget_or_default(name: str, default: str) -> str:
    """Read a Databricks widget when available; otherwise use the default value."""
    utils = _databricks_utils()
    if utils is None:
        return default
    try:
        return utils.widgets.get(name)
    except Exception:
        try:
            utils.widgets.text(name, default)
            return utils.widgets.get(name)
        except Exception:
            return default


def upstream_task_value(task_key: str, name: str, default: str) -> str:
    """Read a job task value, with a fallback for standalone execution."""
    utils = _databricks_utils()
    if utils is None:
        return default
    try:
        return str(
            utils.jobs.taskValues.get(
                taskKey=task_key,
                key=name,
                debugValue=default,
            )
        )
    except Exception:
        return default


def set_task_value(name: str, value: str | bool, *, required: bool = False) -> bool:
    """Publish context for subsequent tasks when running as a job."""
    utils = _databricks_utils()
    if utils is None:
        if required:
            raise RuntimeError(f"DBUtils is unavailable to publish task value: {name}")
        return False
    try:
        utils.jobs.taskValues.set(key=name, value=value)
        return True
    except Exception as exc:
        if required:
            raise RuntimeError(f"Failed to publish task value: {name}") from exc
        return False


CATALOG = _widget_or_default("catalog", "retail_analytics")
STAGING_SCHEMA = _widget_or_default("staging_schema", "default_staging")
INTERMEDIATE_SCHEMA = _widget_or_default("intermediate_schema", "default_intermediate")
MARTS_SCHEMA = _widget_or_default("marts_schema", "default_marts")


def resolve_reference_date(
    value: str | date | None = None,
    *,
    now: datetime | None = None,
) -> date:
    """Resolve the operational date, with a fallback to São Paulo time."""
    if isinstance(value, date):
        return value
    if value and value.strip():
        try:
            return date.fromisoformat(value.strip())
        except ValueError as exc:
            raise ValueError("reference_date must use the YYYY-MM-DD format") from exc
    current = now or datetime.now(ZoneInfo(OPERATIONAL_TIMEZONE))
    if current.tzinfo is None:
        current = current.replace(tzinfo=ZoneInfo(OPERATIONAL_TIMEZONE))
    return current.astimezone(ZoneInfo(OPERATIONAL_TIMEZONE)).date()


def previous_month_end(reference_date: date) -> date:
    """Return the last day of the closed month before the reference date."""
    return reference_date.replace(day=1) - timedelta(days=1)


def shift_month_start(reference: date, months: int) -> date:
    """Shift the first day of a month without external libraries."""
    month_index = reference.year * 12 + reference.month - 1 + months
    return date(month_index // 12, month_index % 12 + 1, 1)


def forecast_horizon_months(cutoff_date: date, months: int = 3) -> tuple[str, ...]:
    """Builds the months immediately following the cutoff."""
    cutoff_month = cutoff_date.replace(day=1)
    return tuple(
        shift_month_start(cutoff_month, offset).strftime("%Y-%m")
        for offset in range(1, months + 1)
    )


def snapshot_id_for(cutoff_date: date) -> str:
    """Stable logical identity for a forecast with a monthly cutoff."""
    return f"forecast_{cutoff_date:%Y_%m}"


def fq_table(schema: str, table: str) -> str:
    """Return a table's fully qualified name in Unity Catalog."""
    return f"{CATALOG}.{schema}.{table}"


@dataclass(frozen=True)
class ForecastConfig:
    """Official demand forecast parameters.

    Eligibility thresholds follow the rule implemented in
    int_monthly_product_location_demand.sql (is_regression_eligible):
    history_month_count >= min_history_months e
    positive_demand_month_count >= min_positive_demand_months.
    """

    reference_date: date
    cutoff_date: date
    horizon_months: tuple[str, ...]
    snapshot_id: str
    execution_id: str
    demand_grain: tuple[str, ...] = (
        "demand_month",
        "product_variant_id",
        "location_id",
    )
    order_status_filter: str = "paid"
    min_history_months: int = 6
    min_positive_demand_months: int = 3


def build_forecast_config(
    reference_date: str | date | None = None,
    *,
    execution_id: str = "local",
    now: datetime | None = None,
) -> ForecastConfig:
    """Creates the complete temporal contract for a run."""
    resolved_reference_date = resolve_reference_date(reference_date, now=now)
    cutoff_date = previous_month_end(resolved_reference_date)
    return ForecastConfig(
        reference_date=resolved_reference_date,
        cutoff_date=cutoff_date,
        horizon_months=forecast_horizon_months(cutoff_date),
        snapshot_id=snapshot_id_for(cutoff_date),
        execution_id=execution_id.strip() or "local",
    )


_REFERENCE_DATE_WIDGET = _widget_or_default("reference_date", "")
_EXECUTION_ID_WIDGET = _widget_or_default("execution_id", "local")
FORECAST_CONFIG = build_forecast_config(
    upstream_task_value("dbt_transformation", "reference_date", _REFERENCE_DATE_WIDGET),
    execution_id=upstream_task_value(
        "dbt_transformation", "execution_id", _EXECUTION_ID_WIDGET
    ),
)
