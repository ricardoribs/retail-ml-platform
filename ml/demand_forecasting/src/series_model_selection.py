"""Auditable selection of the winning product-location baseline.

Selection does not use an aggregate segment metric. Each series must beat
the zero baseline in at least two of the three time windows and reduce
combined WAPE. Without this evidence, zero remains the validated winner."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

MODEL_BASELINE_ZERO = "baseline_zero"
MODEL_NAMES = (
    MODEL_BASELINE_ZERO,
    "baseline_ma3",
    "baseline_ma6",
    "baseline_seasonal",
)
MIN_VALIDATION_FOLDS = 3
MIN_FOLD_WINS = 2
MIN_WAPE_IMPROVEMENT = 2.0


@dataclass(frozen=True)
class SeriesModelDecision:
    """Versionable result of per-series model selection."""

    selected_model_name: str
    validation_status: str
    combined_wape: float | None
    zero_wape: float | None
    wape_improvement: float | None
    validation_folds: int
    folds_won_against_zero: int
    positive_actual_rows: int
    selection_reason: str


def _finite_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _aggregate_model(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float | None, int, int]:
    """Return combined WAPE, number of folds, and positive months."""
    if not rows:
        return None, 0, 0
    fold_ids = {int(row["fold_index"]) for row in rows}
    actual_total = sum(float(row["actual_abs_sum"]) for row in rows)
    error_total = sum(float(row["absolute_error_sum"]) for row in rows)
    positive_rows = sum(int(row["positive_actual_rows"]) for row in rows)
    wape = (error_total / actual_total * 100.0) if actual_total > 0 else None
    return wape, len(fold_ids), positive_rows


def select_series_champion(
    metric_rows: Sequence[Mapping[str, Any]],
) -> SeriesModelDecision:
    """Select one model when there is sufficient temporal evidence.

    ``metric_rows`` must contain one row per model and fold for the same series,
    com ``actual_abs_sum``, ``absolute_error_sum``, ``positive_actual_rows`` e
    ``fold_index``. The zero baseline is never rejected; it is the safe route
    when the other candidates do not show consistent gains.
    """
    by_model: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in metric_rows:
        model_name = str(row["model_name"])
        if model_name not in MODEL_NAMES:
            raise ValueError(f"Unsupported model for selection: {model_name}")
        by_model[model_name].append(row)

    zero_rows = by_model.get(MODEL_BASELINE_ZERO, [])
    zero_wape, zero_folds, positive_rows = _aggregate_model(zero_rows)
    if zero_folds == 0:
        raise ValueError("Per-series selection requires the zero baseline")
    if zero_folds != MIN_VALIDATION_FOLDS:
        return SeriesModelDecision(
            selected_model_name=MODEL_BASELINE_ZERO,
            validation_status="insufficient_validation_signal",
            combined_wape=zero_wape,
            zero_wape=zero_wape,
            wape_improvement=None,
            validation_folds=zero_folds,
            folds_won_against_zero=0,
            positive_actual_rows=positive_rows,
            selection_reason=(
                "Series has fewer than three complete temporal windows; keep the "
                "zero baseline under review until enough evidence is collected."
            ),
        )

    if positive_rows == 0:
        return SeriesModelDecision(
            selected_model_name=MODEL_BASELINE_ZERO,
            validation_status="validated_zero_no_demand",
            combined_wape=None,
            zero_wape=None,
            wape_improvement=None,
            validation_folds=zero_folds,
            folds_won_against_zero=0,
            positive_actual_rows=0,
            selection_reason=(
                "No positive demand was observed in the three validation windows; "
                "the zero baseline was selected with limited confidence."
            ),
        )

    zero_by_fold = {
        int(row["fold_index"]): _finite_or_none(row.get("wape")) for row in zero_rows
    }
    eligible: list[tuple[str, float, int, int]] = []
    for model_name in MODEL_NAMES:
        if model_name == MODEL_BASELINE_ZERO:
            continue
        candidate_rows = by_model.get(model_name, [])
        candidate_wape, folds, _ = _aggregate_model(candidate_rows)
        if (
            candidate_wape is None
            or folds != MIN_VALIDATION_FOLDS
            or zero_folds != MIN_VALIDATION_FOLDS
        ):
            continue
        wins = sum(
            1
            for row in candidate_rows
            if (
                (candidate_fold_wape := _finite_or_none(row.get("wape"))) is not None
                and (zero_fold_wape := zero_by_fold.get(int(row["fold_index"])))
                is not None
                and candidate_fold_wape < zero_fold_wape
            )
        )
        improvement = zero_wape - candidate_wape
        if wins >= MIN_FOLD_WINS and improvement >= MIN_WAPE_IMPROVEMENT:
            eligible.append((model_name, candidate_wape, wins, folds))

    if not eligible:
        validation_status = (
            "validated_zero_sparse" if positive_rows == 1 else "validated_baseline_zero"
        )
        signal_qualifier = (
            "with only one positive occurrence observed"
            if positive_rows == 1
            else "across the three validation windows"
        )
        return SeriesModelDecision(
            selected_model_name=MODEL_BASELINE_ZERO,
            validation_status=validation_status,
            combined_wape=zero_wape,
            zero_wape=zero_wape,
            wape_improvement=0.0,
            validation_folds=zero_folds,
            folds_won_against_zero=0,
            positive_actual_rows=positive_rows,
            selection_reason=(
                "No candidate reduced WAPE by at least 2 percentage points and beat "
                f"zero in two of the three windows {signal_qualifier}; the zero "
                "baseline was selected."
            ),
        )

    model_name, candidate_wape, wins, folds = min(
        eligible, key=lambda item: (item[1], item[0])
    )
    return SeriesModelDecision(
        selected_model_name=model_name,
        validation_status="validated_champion",
        combined_wape=candidate_wape,
        zero_wape=zero_wape,
        wape_improvement=zero_wape - candidate_wape,
        validation_folds=folds,
        folds_won_against_zero=wins,
        positive_actual_rows=positive_rows,
        selection_reason=(
            f"{model_name} beat the zero baseline in {wins} of {folds} windows "
            f"and reduced combined WAPE by {zero_wape - candidate_wape:.2f} percentage points."
        ),
    )
