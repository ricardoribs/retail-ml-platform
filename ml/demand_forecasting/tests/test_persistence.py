"""Pure tests for the metric persistence contract."""

from datetime import date, datetime

import pytest

from ml.demand_forecasting.src.persistence import (
    MetricRunMetadata,
    prepare_metric_records,
)

METADATA = MetricRunMetadata(
    run_id="run-123",
    model_version="cp5_v1",
    train_start_month=date(2020, 7, 1),
    train_end_month=date(2026, 5, 31),
    test_start_month=date(2026, 6, 1),
    test_end_month=date(2026, 8, 31),
    cutoff_date=date(2026, 8, 31),
)


def _metric_row(model_name: str = "ridge") -> dict:
    return {
        "model_name": model_name,
        "model_family": "linear",
        "mae": 2.0,
        "rmse": 3.0,
        "mape": 4.0,
        "wape": 5.0,
        "evaluated_rows": 10,
        "fallback_rows": 0,
        "training_rows": 100,
        "training_seconds": 1.5,
        "parameters_json": '{"regParam": 0.1}',
    }


def test_prepare_metric_records_builds_table_contract():
    created_at = datetime(2026, 9, 8, 12, 0, 0)

    records = prepare_metric_records(
        [_metric_row()],
        METADATA,
        best_model_name="ridge",
        created_at=created_at,
    )

    assert records == [
        {
            "run_id": "run-123",
            "model_name": "ridge",
            "model_family": "linear",
            "model_version": "cp5_v1",
            "evaluation_window": "2026-06-01/2026-08-31",
            "train_start_month": date(2020, 7, 1),
            "train_end_month": date(2026, 5, 31),
            "test_start_month": date(2026, 6, 1),
            "test_end_month": date(2026, 8, 31),
            "cutoff_date": date(2026, 8, 31),
            "mae": 2.0,
            "rmse": 3.0,
            "mape": 4.0,
            "wape": 5.0,
            "evaluated_rows": 10,
            "fallback_rows": 0,
            "training_rows": 100,
            "training_seconds": 1.5,
            "parameters_json": '{"regParam": 0.1}',
            "is_best_model": True,
            "created_at": created_at,
        }
    ]


def test_prepare_metric_records_converts_undefined_percentages_to_null():
    row = _metric_row()
    row["mape"] = float("nan")
    row["wape"] = float("inf")

    record = prepare_metric_records([row], METADATA, "ridge")[0]

    assert record["mape"] is None
    assert record["wape"] is None


@pytest.mark.parametrize("metric_name", ["mae", "rmse"])
def test_prepare_metric_records_rejects_undefined_required_metrics(metric_name):
    row = _metric_row()
    row[metric_name] = float("nan")

    with pytest.raises(ValueError, match=rf"{metric_name}.*finita"):
        prepare_metric_records([row], METADATA, "ridge")


@pytest.mark.parametrize(
    ("rows", "best_model_name", "message"),
    [
        ([], "ridge", "cannot be empty"),
        ([_metric_row("")], "ridge", "model_name"),
        ([_metric_row("ridge"), _metric_row("ridge")], "ridge", "must be unique"),
        ([_metric_row()], "random_forest", "must exist"),
    ],
)
def test_prepare_metric_records_rejects_invalid_contract(
    rows, best_model_name, message
):
    with pytest.raises(ValueError, match=message):
        prepare_metric_records(rows, METADATA, best_model_name)


def test_prepare_metric_records_requires_evaluated_rows():
    row = _metric_row()
    row["evaluated_rows"] = 0

    with pytest.raises(ValueError, match="maior que zero"):
        prepare_metric_records([row], METADATA, "ridge")
