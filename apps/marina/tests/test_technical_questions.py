"""Smoke tests for technical questions."""

from __future__ import annotations

import importlib

import pytest

from apps.marina.src.retriever import RetrievedChunk

marina_app = importlib.import_module("apps.marina.app")


class SmokeMartProvider:
    """Minimal deterministic data for exercising the endpoint locally."""

    catalog = "retail_analytics"
    schema = "default_marts"

    def forecast(self):
        return {"horizons": 3, "series": [{"period": "2026-09", "demand_qty": 10}]}

    def model_metrics(self):
        return [{"model_name": "gradient_boosted_trees", "wape": 176.792}]

    def seasonality(self):
        return {"product_id": 608, "series": [{"month_number": 1}]}

    def seasonality_ranking(self):
        return [{"product_variant_id": 608, "cv": 0.908}]

    def locations(self):
        return [{"location_name": "Loja A", "growth_abs_qty": 20}]

    def recommendations(self):
        return [{"macro_region": "Central-West", "candidate_rank": 1}]


TECHNICAL_SMOKE_CASES = [
    (
        "Qual é a previsão para os próximos três meses?",
        {"agg_location_demand_forecast"},
    ),
    ("Como os modelos foram avaliados?", {"fct_forecast_model_metrics"}),
    ("Quais produtos são mais sazonais?", {"agg_product_seasonality"}),
    ("Como interpretar a sazonalidade do produto 608?", {"agg_product_seasonality"}),
    ("Qual loja apresenta o maior crescimento?", {"agg_location_forecast_growth"}),
    (
        "Qual região é candidata a um novo CD?",
        {"agg_distribution_center_recommendation"},
    ),
    (
        "Devemos abrir um novo centro de distribuição?",
        {"agg_distribution_center_recommendation"},
    ),
    (
        "Quais são os indicadores atuais?",
        {
            "agg_location_demand_forecast",
            "fct_forecast_model_metrics",
            "agg_product_seasonality",
            "agg_location_forecast_growth",
            "agg_distribution_center_recommendation",
        },
    ),
]


@pytest.mark.parametrize(("question", "expected_tables"), TECHNICAL_SMOKE_CASES)
def test_technical_question_returns_answer_and_expected_sources(
    monkeypatch,
    question: str,
    expected_tables: set[str],
) -> None:
    """Send each question to the endpoint and check its answer and sources."""
    chunk = RetrievedChunk(
        source_path="ml/demand_forecasting/README.md",
        source_type="readme",
        heading="Forecast",
        chunk_text="Contexto técnico validado.",
        score=0.9,
    )
    monkeypatch.setattr(marina_app, "mart_provider", SmokeMartProvider())
    monkeypatch.setattr(marina_app.retriever, "retrieve", lambda _question: [chunk])

    def smoke_answer(messages):
        source_marker = (
            "[[SOURCES:"
            + ",".join(
                f"S{index}"
                for index, line in enumerate(
                    messages[1]["content"]
                    .split("Fontes disponíveis:\n", 1)[1]
                    .split("\n\n", 1)[0]
                    .splitlines(),
                    start=1,
                )
                if line.startswith("[S")
            )
            + "]]"
        )
        if question == "Como interpretar a sazonalidade do produto 608?":
            return (
                "confidence=high significa mínimo de três observações em cada mês "
                f"do calendário. {source_marker}"
            )
        return f"Resposta técnica validada pelo smoke test. {source_marker}"

    monkeypatch.setattr(marina_app.llm, "call_llm", smoke_answer)

    response = marina_app.app.test_client().post(
        "/api/chat", json={"question": question}
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "answered"
    if question == "Como interpretar a sazonalidade do produto 608?":
        assert "três observações em cada mês do calendário" in payload["answer"]
    else:
        assert payload["answer"] == "Resposta técnica validada pelo smoke test."
    assert "ml/demand_forecasting/README.md" in payload["sources"]
    for table in expected_tables:
        assert any(source.endswith(f".{table}") for source in payload["sources"])
