"""Testes da selecao temporal de modelo por produto-localidade."""

from ml.demand_forecasting.src.series_model_selection import (
    MODEL_BASELINE_ZERO,
    select_series_champion,
)


def _row(model_name: str, fold_index: int, *, actual: float, error: float):
    return {
        "model_name": model_name,
        "fold_index": fold_index,
        "actual_abs_sum": actual,
        "absolute_error_sum": error,
        "positive_actual_rows": 1 if actual > 0 else 0,
        "wape": error / actual * 100 if actual else None,
    }


def test_selects_candidate_that_wins_two_folds_and_reduces_combined_wape():
    rows = [
        *[
            _row(MODEL_BASELINE_ZERO, fold, actual=10.0, error=10.0)
            for fold in range(1, 4)
        ],
        _row("baseline_ma3", 1, actual=10.0, error=4.0),
        _row("baseline_ma3", 2, actual=10.0, error=5.0),
        _row("baseline_ma3", 3, actual=10.0, error=12.0),
    ]

    decision = select_series_champion(rows)

    assert decision.selected_model_name == "baseline_ma3"
    assert decision.validation_status == "validated_champion"
    assert decision.folds_won_against_zero == 2
    assert decision.wape_improvement == 30.0


def test_keeps_zero_when_candidate_only_wins_one_fold():
    rows = [
        *[
            _row(MODEL_BASELINE_ZERO, fold, actual=10.0, error=10.0)
            for fold in range(1, 4)
        ],
        _row("baseline_seasonal", 1, actual=10.0, error=4.0),
        _row("baseline_seasonal", 2, actual=10.0, error=12.0),
        _row("baseline_seasonal", 3, actual=10.0, error=12.0),
    ]

    decision = select_series_champion(rows)

    assert decision.selected_model_name == MODEL_BASELINE_ZERO
    assert decision.validation_status == "validated_baseline_zero"


def test_selects_zero_when_one_observed_sale_is_not_beaten_by_candidates():
    rows = [
        _row(MODEL_BASELINE_ZERO, 1, actual=0.0, error=0.0),
        _row(MODEL_BASELINE_ZERO, 2, actual=10.0, error=10.0),
        _row(MODEL_BASELINE_ZERO, 3, actual=0.0, error=0.0),
        _row("baseline_ma6", 1, actual=0.0, error=0.0),
        _row("baseline_ma6", 2, actual=10.0, error=3.0),
        _row("baseline_ma6", 3, actual=0.0, error=0.0),
    ]

    decision = select_series_champion(rows)

    assert decision.selected_model_name == MODEL_BASELINE_ZERO
    assert decision.validation_status == "validated_zero_sparse"


def test_selects_zero_when_no_positive_demand_occurs_in_three_folds():
    rows = [
        *[
            _row(MODEL_BASELINE_ZERO, fold, actual=0.0, error=0.0)
            for fold in range(1, 4)
        ],
        *[_row("baseline_ma3", fold, actual=0.0, error=1.0) for fold in range(1, 4)],
    ]

    decision = select_series_champion(rows)

    assert decision.selected_model_name == MODEL_BASELINE_ZERO
    assert decision.validation_status == "validated_zero_no_demand"
    assert decision.validation_folds == 3


def test_keeps_zero_in_review_when_a_series_does_not_have_three_folds():
    rows = [
        _row(MODEL_BASELINE_ZERO, 1, actual=10.0, error=10.0),
        _row(MODEL_BASELINE_ZERO, 2, actual=10.0, error=10.0),
        _row("baseline_ma3", 1, actual=10.0, error=2.0),
        _row("baseline_ma3", 2, actual=10.0, error=2.0),
    ]

    decision = select_series_champion(rows)

    assert decision.selected_model_name == MODEL_BASELINE_ZERO
    assert decision.validation_status == "insufficient_validation_signal"
