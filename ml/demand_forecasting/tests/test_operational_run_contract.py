"""Static tests for the operational contract across dbt, jobs, and notebooks."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
JOB_PATH = REPO_ROOT / "ingestion" / "bundle" / "bronze_ingestion_job.yml"
DBT_PROJECT_PATH = REPO_ROOT / "dbt" / "dbt_project.yml"
DBT_RUNNER_PATH = REPO_ROOT / "dbt" / "notebooks" / "dbt_runner.py"
MONTHLY_DEMAND_PATH = (
    REPO_ROOT
    / "dbt"
    / "models"
    / "intermediate"
    / "int_monthly_product_location_demand.sql"
)
CUTOFF_TEST_PATH = (
    REPO_ROOT
    / "dbt"
    / "tests"
    / "business_logic"
    / "test_monthly_product_location_demand_respects_cutoff.sql"
)
NOTEBOOK_DIR = REPO_ROOT / "ml" / "demand_forecasting" / "notebooks"
SQL_DIR = REPO_ROOT / "ml" / "demand_forecasting" / "sql"
INVENTORY_REFERENCE_PATHS = (
    REPO_ROOT / "dbt" / "models" / "intermediate" / "int_daily_demand.sql",
    REPO_ROOT / "dbt" / "models" / "intermediate" / "int_inventory_demand_metrics.sql",
    REPO_ROOT / "dbt" / "models" / "intermediate" / "int_open_purchases.sql",
    REPO_ROOT / "dbt" / "models" / "marts" / "fct_inventory_replenishment_snapshot.sql",
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_job_exposes_reference_date_and_execution_id():
    job = _read(JOB_PATH)

    assert "- name: reference_date" in job
    assert "- name: execution_id" in job
    assert 'default: "{{job.run_id}}"' in job
    assert "base_parameters:" not in job
    top_level_tasks = [
        line for line in job.splitlines() if line.startswith("        - task_key:")
    ]
    assert len(top_level_tasks) == 17


def test_baseline_is_not_a_required_job_task():
    job = _read(JOB_PATH)

    assert "task_key: demand_forecast_baseline" not in job
    assert "task_key: demand_forecast_train_evaluate" in job


def test_spark_ml_tasks_use_supported_serverless_environment():
    job = _read(JOB_PATH)
    training_task = job.split(
        "        - task_key: demand_forecast_train_evaluate", maxsplit=1
    )[1].split("\n        - task_key:", maxsplit=1)[0]
    generation_task = job.split(
        "        - task_key: demand_forecast_generate", maxsplit=1
    )[1].split("\n        - task_key:", maxsplit=1)[0]
    selection_task = job.split(
        "        - task_key: forecast_series_model_selection", maxsplit=1
    )[1].split("\n        - task_key:", maxsplit=1)[0]
    forecast_environment = job.split("- environment_key: forecast_ml", maxsplit=1)[-1]

    assert "environment_key: forecast_ml" in training_task
    assert "environment_key: forecast_ml" in generation_task
    assert "environment_key: forecast_ml" in selection_task
    assert 'environment_version: "4"' in forecast_environment


def test_dbt_cutoff_is_derived_instead_of_fixed():
    project = _read(DBT_PROJECT_PATH)
    runner = _read(DBT_RUNNER_PATH)
    monthly_demand = _read(MONTHLY_DEMAND_PATH)
    cutoff_test = _read(CUTOFF_TEST_PATH)

    assert 'reference_date_sql: "CURRENT_DATE"' in project
    assert "2026-08-31" not in project
    assert '"reference_date_sql"' in runner
    assert 'set_task_value("reference_date"' in runner
    assert "forecast_cutoff_date()" in monthly_demand
    assert "cast(s.sale_datetime as date) <= p.cutoff_date" in monthly_demand
    assert "demand.demand_month > p.cutoff_date" in cutoff_test


def test_inventory_and_forecast_share_the_operational_reference_date():
    for path in INVENTORY_REFERENCE_PATHS:
        assert "var('forecast_config')['reference_date_sql']" in _read(path)


def test_forecast_uses_stable_snapshot_and_exact_model_run():
    training = _read(NOTEBOOK_DIR / "02_train_evaluate_models.py")
    forecast = _read(NOTEBOOK_DIR / "03_generate_forecast.py")

    assert 'MODEL_RUN_ID = f"model_{FORECAST_CONFIG.snapshot_id}"' in training
    assert 'set_task_value("model_run_id", MODEL_RUN_ID)' in training
    assert "FORECAST_RUN_ID = FORECAST_CONFIG.snapshot_id" in forecast
    assert "SERIES_SELECTION_RUN_ID = upstream_task_value(" in forecast
    assert 'F.col("run_id") == F.lit(SERIES_SELECTION_RUN_ID)' in forecast
    assert '.orderBy(F.col("created_at").desc())' not in forecast
    assert "whenNotMatchedBySourceDelete" in forecast


def test_growth_rankings_require_a_validated_aggregate_forecast():
    job = _read(JOB_PATH)
    forecast = _read(NOTEBOOK_DIR / "03_generate_forecast.py")
    aggregate_forecast = _read(
        NOTEBOOK_DIR / "20_generate_location_aggregate_forecast.py"
    )

    gate_task = job.split(
        "        - task_key: forecast_actionability_gate", maxsplit=1
    )[1].split("\n        - task_key:", maxsplit=1)[0]
    location_task = job.split(
        "        - task_key: location_forecast_growth", maxsplit=1
    )[1].split("\n        - task_key:", maxsplit=1)[0]

    assert '"has_actionable_forecast"' in forecast
    assert "required=True" in forecast
    reuse_block = forecast.split(
        "if not MODEL_REFRESHED and can_reuse_forecast:", maxsplit=1
    )[1].split("# COMMAND ----------", maxsplit=1)[0]
    assert '"has_actionable_forecast"' in reuse_block
    assert "fct_forecast_series_publication" not in reuse_block
    assert "PUBLICATION_TABLE" in reuse_block
    assert "condition_task:" in gate_task
    assert "has_actionable_aggregate_forecast" in gate_task
    assert 'outcome: "true"' in location_task
    assert "location_aggregate_forecast" in gate_task
    assert (
        'set_task_value("has_actionable_aggregate_forecast", True' in aggregate_forecast
    )
    assert (
        "CUTOFF_MONTH = FORECAST_CONFIG.cutoff_date.replace(day=1)"
        in aggregate_forecast
    )


def test_retroactive_location_validation_is_read_only_and_out_of_sample():
    notebook = _read(NOTEBOOK_DIR / "21_location_forecast_retrospective_validation.py")

    assert "agg_location_demand_forecast" in notebook
    assert "fct_sales" in notebook
    assert 'F.col("forecast_month").isin(*FORECAST_MONTHS)' in notebook
    assert 'F.to_date("sales.sale_datetime") > F.lit(CUTOFF_DATE)' in notebook
    assert "monthly_total" in notebook
    assert "month_location" in notebook
    assert "saveAsTable" not in notebook
    assert ".write." not in notebook


def test_replenishment_policy_runs_even_when_forecast_rankings_are_blocked():
    job = _read(JOB_PATH)
    notebook = _read(NOTEBOOK_DIR / "17_replenishment_planning_policy.py")

    replenishment_task = job.split(
        "        - task_key: replenishment_planning_policy", maxsplit=1
    )[1].split("\n        - task_key:", maxsplit=1)[0]

    assert "task_key: demand_forecast_generate" in replenishment_task
    assert "forecast_actionability_gate" not in replenishment_task
    assert "fct_inventory_replenishment_snapshot" in notebook
    assert '"dbt_transformation",' in notebook
    assert "inventory_reference_date = INVENTORY_REFERENCE_DATE" in notebook
    assert '"forecast_zero_not_used": True' in notebook
    assert '"manual_approval_required": True' in notebook


def test_downstream_notebooks_use_generated_snapshot():
    for notebook_name in (
        "04_location_forecast_growth.py",
        "06_cd_forecast_growth.py",
        "07_distribution_center_recommendation.py",
    ):
        notebook = _read(NOTEBOOK_DIR / notebook_name)
        assert '"location_aggregate_forecast", "snapshot_id"' in notebook
        assert "latest_complete_run" not in notebook
        assert 'orderBy(F.col("run_created_at").desc()' not in notebook


def test_cd_growth_persists_the_global_forecast_period():
    notebook = _read(NOTEBOOK_DIR / "06_cd_forecast_growth.py")

    assert "FORECAST_START_MONTH = date.fromisoformat(" in notebook
    assert "FORECAST_END_MONTH = date.fromisoformat(" in notebook
    assert (
        'F.lit(FORECAST_START_MONTH).cast("date").alias("forecast_start_month")'
        in notebook
    )
    assert (
        'F.lit(FORECAST_END_MONTH).cast("date").alias("forecast_end_month")' in notebook
    )
    assert 'F.lit(None).cast("date").alias("forecast_start_month")' not in notebook
    assert 'F.lit(None).cast("date").alias("forecast_end_month")' not in notebook


def test_job_notebooks_add_repository_root_before_local_imports():
    for notebook_name in (
        "02_train_evaluate_models.py",
        "03_generate_forecast.py",
        "04_location_forecast_growth.py",
        "05_seasonality_ranking.py",
        "15_series_policy.py",
        "16_segmented_baseline_backtest.py",
        "18_series_model_selection.py",
        "20_generate_location_aggregate_forecast.py",
        "06_cd_forecast_growth.py",
        "07_distribution_center_recommendation.py",
    ):
        notebook = _read(NOTEBOOK_DIR / notebook_name)
        path_bootstrap = notebook.index("sys.path.insert(0, str(_REPO_ROOT))")
        project_import = notebook.index("from ml.demand_forecasting")

        assert path_bootstrap < project_import


def test_occurrence_signal_audit_is_manual_read_only_and_as_of_cutoff():
    notebook = _read(NOTEBOOK_DIR / "10_audit_occurrence_signal.py")
    job = _read(JOB_PATH)

    assert '"2025-11-30,2026-08-31"' in notebook
    assert 'F.col("demand_month") <= F.lit(split.train_end)' in notebook
    assert 'F.add_months("demand_month", -12)' in notebook
    assert "10_audit_occurrence_signal" not in job
    for write_operation in (".write", "saveAsTable", "DeltaTable", "MERGE INTO"):
        assert write_operation not in notebook


def test_hierarchical_backtest_is_manual_read_only_and_as_of_cutoff():
    notebook = _read(NOTEBOOK_DIR / "11_hierarchical_backtest.py")
    job = _read(JOB_PATH)

    assert '"2025-11-30,2026-08-31"' in notebook
    assert 'F.col("demand_month") <= F.lit(split.train_end)' in notebook
    assert '"category_location_wape"' in notebook
    assert '"sku_location_wape"' in notebook
    assert "11_hierarchical_backtest" not in job
    for write_operation in (".write", "saveAsTable", "DeltaTable", "MERGE INTO"):
        assert write_operation not in notebook


def test_inventory_availability_audit_is_manual_and_read_only():
    notebook = _read(NOTEBOOK_DIR / "12_audit_inventory_availability.py")
    job = _read(JOB_PATH)

    assert '"2025-11-30,2026-08-31"' in notebook
    assert 'fq_table(STAGING_SCHEMA, "stg_stock_levels")' in notebook
    assert 'fq_table(STAGING_SCHEMA, "stg_stock_movements")' in notebook
    assert "12_audit_inventory_availability" not in job
    for write_operation in (".write", "saveAsTable", "DeltaTable", "MERGE INTO"):
        assert write_operation not in notebook


def test_intermittent_backtest_is_manual_read_only_and_as_of_cutoff():
    notebook = _read(NOTEBOOK_DIR / "14_intermittent_bayes_backtest.py")
    job = _read(JOB_PATH)

    assert '"2025-11-30,2026-08-31"' in notebook
    assert 'F.col("demand_month") <= F.lit(split.train_end)' in notebook
    assert '"store_all"' in notebook
    assert '"sku_location_wape"' in notebook
    assert "14_intermittent_bayes_backtest" not in job
    for write_operation in (".write", "saveAsTable", "DeltaTable", "MERGE INTO"):
        assert write_operation not in notebook


def test_sql_warehouse_backtest_is_read_only_and_checks_both_cutoffs():
    query = _read(SQL_DIR / "14_intermittent_bayes_backtest.sql").lower()

    assert "date '2025-11-30'" in query
    assert "date '2026-08-31'" in query
    assert "retail_analytics.default_intermediate.int_forecast_feature_matrix" in query
    assert "passes_all_store_periods" in query
    for write_operation in (
        "insert ",
        "update ",
        "delete ",
        "merge ",
        "create ",
        "replace ",
        "drop ",
    ):
        assert write_operation not in query
