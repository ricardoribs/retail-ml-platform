"""Tests for Marina's interface and metrics endpoint."""

from __future__ import annotations

import importlib
import logging
import subprocess
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest

from apps.marina.src.retriever import RetrievedChunk

marina_app = importlib.import_module("apps.marina.app")
APP_ROOT = Path(__file__).resolve().parents[1]


def _chunk(source_path="readme.md"):
    return RetrievedChunk(
        source_path=source_path,
        source_type="readme",
        heading="Visão geral",
        chunk_text="Distribution Center 1 lidera o ranking de crescimento.",
        score=0.72,
    )


class FakeMartProvider:
    def visuals(self):
        return {
            "status": "ready",
            "forecast": {
                "horizons": 3,
                "series": [{"period": "2026-10", "demand_qty": 12.5}],
            },
            "seasonality": {
                "product_id": 608,
                "sku": "SKU-462189",
                "product_name": "Outboard Motor 2640",
                "cv": 0.908,
                "series": [
                    {
                        "product_variant_id": 608,
                        "sku": "SKU-462189",
                        "product_name": "Outboard Motor 2640",
                        "month_number": 1,
                        "seasonality_index": 1.2,
                    }
                ],
            },
            "locations": [{"location_name": "Loja 1", "growth_abs_qty": 22.0}],
            "distribution_centers": [
                {"location_name": "Distribution Center 1", "growth_abs_qty": 18.0}
            ],
            "recommendations": [
                {
                    "macro_region": "Central-West",
                    "candidate_rank": 1,
                    "recommendation_score": 1.0,
                }
            ],
            "top_candidate": {"macro_region": "Central-West"},
        }


class EmptyMartProvider:
    catalog = "retail_analytics"
    schema = "default_marts"

    def visuals(self):
        return {}

    def forecast(self):
        return {"horizons": 0, "series": []}

    def forecast_publication(self):
        return {}

    def model_metrics(self):
        return []

    def seasonality(self):
        return {"product_id": 608, "cv": None, "series": []}

    def seasonality_ranking(self):
        return []

    def locations(self):
        return []

    def distribution_centers(self):
        return []

    def recommendations(self):
        return []


@pytest.fixture(autouse=True)
def isolate_mart_provider(monkeypatch):
    """Impede que testes unitários consultem o warehouse configurado."""
    monkeypatch.setattr(marina_app, "mart_provider", EmptyMartProvider())


def test_visuals_endpoint_returns_mart_payload(monkeypatch):
    monkeypatch.setattr(marina_app, "mart_provider", FakeMartProvider())
    response = marina_app.app.test_client().get("/api/visuals")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] == "ready"
    assert payload["forecast"]["horizons"] == 3
    assert payload["seasonality"]["product_name"] == "Outboard Motor 2640"
    assert payload["distribution_centers"][0]["location_name"] == "Distribution Center 1"
    assert payload["top_candidate"]["macro_region"] == "Central-West"


def test_local_demo_provider_returns_complete_visual_payload():
    payload = marina_app.LocalDemoMartProvider().visuals()

    assert payload["status"] == "ready"
    assert payload["data_mode"] == "local_demo"
    assert len(payload["forecast"]["series"]) == 3
    assert payload["forecast"]["location_count"] == 6
    assert payload["forecast"]["store_count"] == 3
    assert payload["forecast"]["distribution_center_count"] == 3
    assert len(payload["seasonality"]["series"]) == 12
    assert (
        payload["seasonality"]["selection_method"]
        == "highest_cv_minimum_monthly_observations"
    )
    assert payload["locations"]
    assert payload["distribution_centers"]
    assert payload["locations"][0]["actual_start_month"] == "2026-06-01"
    assert payload["locations"][0]["forecast_end_month"] == "2026-11-01"
    assert payload["top_candidate"]["candidate_rank"] == 1


def test_visuals_endpoint_degrades_without_marts(monkeypatch):
    class UnavailableProvider:
        def visuals(self):
            raise RuntimeError("sem conexão")

    monkeypatch.setattr(marina_app, "mart_provider", UnavailableProvider())
    response = marina_app.app.test_client().get("/api/visuals")

    assert response.status_code == 200
    assert response.get_json()["status"] == "unavailable"


def test_home_contains_technical_visuals():
    response = marina_app.app.test_client().get("/")

    assert response.status_code == 200
    assert b"Indicadores" in response.data
    assert b"/api/visuals" in response.data
    assert b"message-sources" in response.data
    assert b"payload.sources" in response.data
    assert b"renderForecastChart" in response.data
    assert b"renderSeasonalityChart" in response.data
    assert b"renderGrowth" in response.data
    assert b"distribution-centers-visual" in response.data
    assert (
        "Ranking por demanda total prevista nos próximos 3 meses".encode()
        in response.data
    )
    assert b"visual-card-top" not in response.data


def test_home_uses_retail_analytics_logo_as_favicon():
    client = marina_app.app.test_client()

    response = client.get("/")
    assert response.status_code == 200
    assert (
        b'<link rel="icon" href="/static/logo-retail-analytics.svg" type="image/svg+xml">'
        in response.data
    )

    favicon = client.get("/favicon.ico")
    assert favicon.status_code == 200
    assert favicon.mimetype == "image/svg+xml"


def test_home_uses_a_z_shaped_operational_layout():
    response = marina_app.app.test_client().get("/")
    overview_intro = "Da previsão de demanda à recomendação logística".encode()

    assert "Indicadores operacionais".encode() in response.data
    assert overview_intro not in response.data
    assert b"overview-header" in response.data
    assert b"marina-launch-button" in response.data
    assert b"visual-card--forecast" in response.data
    assert b"visual-card--seasonality" in response.data
    assert b"visual-card--growth" in response.data
    assert b"visual-card--cd-growth" in response.data
    assert b"visual-card--recommendation" in response.data
    assert b"grid-template-columns: repeat(12, minmax(0, 1fr));" in response.data
    assert b"grid-column: 1 / -1;" in response.data
    assert b"grid-template-columns: minmax(0, 1fr) 250px;" in response.data
    forecast_header = (
        b".visual-card--forecast .visual-card-header { display: contents; }"
    )
    responsive_width = b"Math.round(element.getBoundingClientRect().width || 308)"
    assert forecast_header in response.data
    assert responsive_width in response.data
    assert b"indicator-result is-unavailable" in response.data
    assert b"function setIndicatorResult(id, text, available)" in response.data
    assert b'class="insights"' not in response.data
    assert b"data-open-marina" in response.data


def test_home_uses_a_professional_visualization_palette():
    response = marina_app.app.test_client().get("/")

    assert b"--chart-navy: #173f5f" in response.data
    assert b"--chart-teal: #287a78" in response.data
    assert b"--chart-amber: #c28a3d" in response.data
    assert b"fill: url(#forecast-area-gradient)" in response.data
    assert b'id="forecast-area-gradient"' in response.data
    assert b".seasonality-bar-high { fill: var(--chart-teal); }" in response.data
    growth_gradient = b"linear-gradient(90deg, var(--chart-navy), var(--chart-teal))"
    assert growth_gradient in response.data


def test_home_renders_location_growth_as_a_proportional_ranking():
    response = marina_app.app.test_client().get("/")
    growth_caption = (
        "Aumento em unidades: previsão futura versus demanda realizada anterior"
    ).encode()

    assert growth_caption in response.data
    assert b"function setGrowthCaption" in response.data
    assert b'id="locations-caption"' in response.data
    assert b'id="distribution-centers-caption"' in response.data
    assert b".growth-rank" in response.data
    assert b".growth-track { display: block;" in response.data
    assert b".growth-fill { display: block;" in response.data
    assert b"ranked.map((row, index)" in response.data
    assert b'class="growth-list" role="list"' in response.data
    assert b'class="growth-row" role="listitem"' in response.data
    assert b'id="indicator-location"' not in response.data


def test_home_explains_operational_indicators_without_ambiguous_labels():
    response = marina_app.app.test_client().get("/")

    assert "Não é uma previsão por localidade, nem acumulada".encode() in response.data
    assert "Oscilação entre meses".encode() in response.data
    assert b"seasonality-legend" in response.data
    assert b'id="indicator-seasonality-selection"' in response.data
    assert b"payload.seasonality.selection_description" in response.data
    assert b"shortMonthNames" in response.data
    assert "Leia os meses na base, da esquerda para a direita".encode() in response.data
    assert "O número acima de cada ponto é o total previsto".encode() in response.data
    assert "Um ponto mais alto indica maior demanda prevista".encode() in response.data
    assert "não representa uma loja ou CD isolado".encode() in response.data
    assert b"unidades previstas" in response.data
    assert b'id="indicator-region"' not in response.data


def test_home_separates_store_and_distribution_center_growth_rankings():
    response = marina_app.app.test_client().get("/")

    assert "Lojas com maior aumento previsto".encode() in response.data
    assert "CDs com maior aumento previsto".encode() in response.data
    assert b'id="locations-visual"' in response.data
    assert b'id="distribution-centers-visual"' in response.data
    assert b'renderGrowth(payload.locations, "locations-visual")' in response.data
    assert (
        b"renderGrowth(payload.distribution_centers, "
        b'"distribution-centers-visual")' in response.data
    )


def test_home_does_not_embed_mock_visuals_or_fake_operational_data():
    response = marina_app.app.test_client().get("/")

    assert response.status_code == 200
    assert "dados simulados".encode() not in response.data
    assert b"mockVisualPayload" not in response.data
    assert b'get("mock")' not in response.data
    assert b"12510" not in response.data
    assert b"Armaz\xc3\xa9m Camargo" not in response.data


def test_home_only_shows_sources_returned_with_each_answer():
    response = marina_app.app.test_client().get("/")

    assert b"marina-modal-references" not in response.data
    assert "Documentos de referência".encode() not in response.data
    assert b"message-sources" in response.data
    assert b"payload.sources" in response.data


def test_home_explains_forecast_horizon_without_hardcoded_periods():
    response = marina_app.app.test_client().get("/")

    assert "Previsão mensal para toda a rede".encode() in response.data
    assert "Escopo do gráfico".encode() in response.data
    description = "O número acima de cada ponto é o total previsto"
    assert description.encode() in response.data
    assert b"renderForecastIndicator(payload.forecast)" in response.data
    assert b"setForecastIndicatorUnavailable()" in response.data
    assert (
        b"`Previs\xc3\xa3o mensal \xe2\x80\x94 ${periods.length} meses`"
        in response.data
    )
    assert b'class="horizon-arrow"' in response.data
    assert "Previsão independente".encode() not in response.data
    assert ">Horizonte<".encode() not in response.data
    assert "Mês 1".encode() not in response.data


def test_home_does_not_show_a_product_before_seasonality_data_is_loaded():
    response = marina_app.app.test_client().get("/")

    assert response.status_code == 200
    assert b'id="indicator-product">Produto aguardando dados</strong>' in response.data
    assert (
        b'setIndicatorResult("indicator-product", productLabel, hasSeasonality)'
        in response.data
    )
    assert (
        b'setIndicatorResult("indicator-product", "Produto indispon\xc3\xadvel", false)'
        in response.data
    )


@pytest.mark.parametrize(
    "asset_name",
    [
        "marina-answer-ok.png",
        "marina-thinking.png",
        "marina-no-answer.png",
        "marina-wave.png",
    ],
)
def test_marina_state_avatar_asset_is_served(asset_name):
    response = marina_app.app.test_client().get(f"/static/{asset_name}")

    assert response.status_code == 200
    assert response.mimetype == "image/png"


def test_home_maps_chat_statuses_to_marina_avatars():
    response = marina_app.app.test_client().get("/")

    assert b'answered: "/static/marina-answer-ok.png"' in response.data
    assert b'no_context: "/static/marina-no-answer.png"' in response.data
    assert b'refused: "/static/marina-no-answer.png"' in response.data
    assert b'unavailable: "/static/marina-no-answer.png"' in response.data
    assert b"/static/marina-thinking.png" in response.data
    assert b"payload.sources || [], payload.status" in response.data
    thinking_label = 'setAttribute("aria-label", "Marina está pensando")'
    assert thinking_label.encode() in response.data


def test_home_aligns_user_messages_to_the_right():
    response = marina_app.app.test_client().get("/")

    assert b".message.user {" in response.data
    assert b"align-self: flex-end;" in response.data
    assert b"text-align: right;" in response.data
    assert b"border-radius: 14px 0 14px 14px;" in response.data


def test_home_places_marina_action_below_topics_and_centers_the_modal():
    response = marina_app.app.test_client().get("/")

    assert b"Marina AI" in response.data
    assert "Pergunte sobre os dados".encode() in response.data
    assert "Temas de análise".encode() in response.data
    assert b".sidebar .marina-launch-button {" in response.data
    assert b'class="marina-launch-button"' in response.data
    assert b"overview-chat-button" not in response.data
    assert b'class="marina-fab"' not in response.data
    assert b"align-items: center; justify-content: center;" in response.data
    assert b"backdrop-filter: blur(6px);" in response.data
    assert b"width: min(520px, 100%);" in response.data


def test_home_keeps_keyboard_focus_inside_the_open_marina_dialog():
    response = marina_app.app.test_client().get("/")

    assert response.status_code == 200
    assert b'aria-labelledby="marina-modal-title"' in response.data
    assert b'event.key === "Escape"' in response.data
    assert b'event.key !== "Tab"' in response.data
    assert b"setModalBackgroundInert(true)" in response.data
    assert b"setModalBackgroundInert(false)" in response.data
    assert b"lastFocusedElement.focus({ preventScroll: true })" in response.data


def test_home_uses_the_actual_chat_classes_in_the_mobile_layout():
    response = marina_app.app.test_client().get("/")

    assert response.status_code == 200
    assert b".marina-composer { padding-left: 12px; }" in response.data
    assert b".marina-send-button { width: 44px; padding: 11px; }" in response.data
    assert b".composer { padding-left: 12px; }" not in response.data
    assert b".send-button { width: 44px; padding: 11px; }" not in response.data


def test_home_preserves_marina_visual_identity_in_the_sidebar_action():
    response = marina_app.app.test_client().get("/")

    assert b"--marina-plum: #51284f;" in response.data
    assert b"--marina-rose: #a64670;" in response.data
    assert b"--marina-peach: #e8b88a;" in response.data
    assert response.data.count(b"background: var(--marina-gradient);") >= 2
    assert b"border-radius: 999px;" in response.data
    assert b"/static/marina-wave.png" in response.data
    assert b"marina-launch-avatar" in response.data
    assert b"width: 36px;" in response.data
    assert b"marina-launch-title" in response.data
    assert b"marina-launch-subtitle" in response.data


def test_home_uses_marina_button_palette_across_the_chat_dialog():
    response = marina_app.app.test_client().get("/")

    assert response.status_code == 200
    assert b"--marina-gradient: linear-gradient(135deg" in response.data
    assert b"--marina-message-gradient: linear-gradient(135deg" in response.data
    assert b"box-shadow: 0 10px 44px rgba(81,40,79,.38);" in response.data
    assert response.data.count(b"background: var(--marina-message-gradient);") >= 3
    assert b"background: rgba(81,40,79,.22);" in response.data


def test_home_only_italicizes_marina_name_in_the_dialog_title():
    response = marina_app.app.test_client().get("/")

    assert response.status_code == 200
    assert "Olá, eu sou a <em>Marina AI</em>.".encode() in response.data
    assert b".marina-modal-title em { font-style: italic; }" in response.data
    assert b".marina-modal-title {" in response.data
    assert b"font-style: normal;" in response.data


def test_home_uses_marina_identity_across_the_entire_sidebar():
    response = marina_app.app.test_client().get("/")

    gradient = b"linear-gradient(135deg, #1a4a7a 0%, #4d8fc4 58%, #e8b88a 100%)"
    assert gradient in response.data
    assert b".sidebar {" in response.data
    assert b"filter: brightness(0) invert(1);" in response.data
    assert b"overflow: visible;" in response.data
    assert b".nav-button:hover .nav-button-icon { stroke: white; }" in response.data


def test_home_uses_layered_marina_surfaces_for_navigation_topics():
    response = marina_app.app.test_client().get("/")

    topic_gradient = (
        b"linear-gradient(115deg, rgba(17,58,89,.82) 0%, "
        b"rgba(55,115,160,.72) 72%, rgba(219,161,111,.5) 100%)"
    )
    assert topic_gradient in response.data
    assert b"border-radius: 18px;" in response.data
    assert b"color: white;" in response.data
    assert b"stroke: rgba(255,255,255,.82);" in response.data
    assert b"transform: translateX(2px);" in response.data


def test_home_keeps_the_composer_visible_for_follow_up_questions():
    response = marina_app.app.test_client().get("/")

    assert b"height: min(640px, calc(100svh - 48px));" in response.data
    assert b".marina-chat-log" in response.data
    assert b"flex: 1; overflow-y: auto;" in response.data
    assert b".marina-composer-wrap {" in response.data
    assert b"flex-shrink: 0;" in response.data
    assert b"question.focus({ preventScroll: true })" in response.data


def test_home_scrolls_the_message_history_after_rendering():
    response = marina_app.app.test_client().get("/")

    assert b"overscroll-behavior: contain;" in response.data
    assert b"scrollbar-gutter: stable;" in response.data
    assert b"function scrollMessagesToBottom()" in response.data
    assert b"requestAnimationFrame(() =>" in response.data
    assert b"messages.scrollTop = messages.scrollHeight;" in response.data


def test_home_only_shows_a_styled_scrollbar_for_long_questions():
    response = marina_app.app.test_client().get("/")

    assert b'question.style.height = "auto";' in response.data
    assert b'question.scrollHeight > maxHeight ? "auto" : "hidden"' in response.data
    assert b"textarea::-webkit-scrollbar-button { display: none; }" in response.data
    assert b"scrollbar-width: thin;" in response.data


def test_home_has_clickable_suggested_questions():
    response = marina_app.app.test_client().get("/")

    assert b"data-q=" in response.data
    assert "Temas de análise".encode() in response.data
    assert "Pergunta preenchida. Revise e pressione Enviar.".encode() in response.data
    assert b"aria-pressed" not in response.data


def test_home_conversation_section_has_accessible_name():
    response = marina_app.app.test_client().get("/")

    assert (
        b'aria-label="Pergunte para a Marina"' in response.data
        or b"aria-labelledby=" in response.data
    )


def test_home_respects_reduced_motion_preference():
    response = marina_app.app.test_client().get("/")

    assert b"prefers-reduced-motion" in response.data
    assert b"animation: none !important" in response.data


def test_home_blocks_concurrent_questions_and_exposes_busy_state():
    response = marina_app.app.test_client().get("/")

    assert b'aria-busy="false"' in response.data
    assert b"let isSubmitting = false" in response.data
    assert b"send.disabled = isSubmitting || !question.value.trim()" in response.data
    assert b"if (isSubmitting)" in response.data
    assert (
        "Aguarde a resposta atual antes de enviar outra pergunta.".encode()
        in response.data
    )


def test_home_uses_overview_label_and_visual_grid_name():
    response = marina_app.app.test_client().get("/")

    assert ">Visão geral<".encode() in response.data
    assert b'class="visuals-grid"' in response.data
    assert b".section-label {" in response.data
    assert b"font-size: 28px;" in response.data
    assert b".starting { margin-top: 0; }" in response.data
    assert b"prompt-grid" not in response.data


def test_home_displays_a_centered_prominent_brand_logo():
    response = marina_app.app.test_client().get("/")

    assert b".brand-company {" in response.data
    assert b"place-items: center;" in response.data
    assert b"width: 118px;" in response.data
    assert b"height: 118px;" in response.data
    assert b".brand-head { position: static; }" in response.data


def test_home_does_not_repeat_branding_in_a_sidebar_footer():
    response = marina_app.app.test_client().get("/")

    assert b"sidebar-footer" not in response.data
    assert "Inteligência de demanda".encode() not in response.data


def test_home_uses_mvp_questions_supported_by_the_available_context():
    response = marina_app.app.test_client().get("/")

    assert "Previsão mensal para toda a rede".encode() in response.data
    assert (
        "Como interpretar a sazonalidade do produto em destaque?".encode()
        in response.data
    )
    assert "Qual modelo teve melhor desempenho?".encode() not in response.data
    assert "Quais produtos apresentaram sazonalidade?".encode() not in response.data
    assert "product_name".encode() in response.data


def test_app_imports_from_deployment_working_directory():
    result = subprocess.run(
        [sys.executable, "-c", "import app"],
        cwd=APP_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_chat_returns_answer_and_sources(monkeypatch):
    monkeypatch.setattr(marina_app.retriever, "retrieve", lambda _question: [_chunk()])
    monkeypatch.setattr(
        marina_app.llm,
        "call_llm",
        lambda _messages: (
            "Distribution Center 1 apresenta o maior crescimento. [[SOURCES:S1]]"
        ),
    )

    response = marina_app.app.test_client().post(
        "/api/chat", json={"question": "Qual loja mais cresceu?"}
    )

    assert response.status_code == 200
    assert response.get_json() == {
        "status": "answered",
        "answer": "Distribution Center 1 apresenta o maior crescimento.",
        "sources": ["readme.md"],
        "question": "Qual loja mais cresceu?",
    }


@pytest.mark.parametrize(
    "generated_answer",
    [
        "Resposta sem marcador.",
        "Resposta com fonte inexistente. [[SOURCES:S99]]",
        "[[SOURCES:S1]]",
    ],
)
def test_chat_never_returns_answered_without_valid_source(
    monkeypatch, generated_answer
):
    monkeypatch.setattr(marina_app.retriever, "retrieve", lambda _question: [_chunk()])
    monkeypatch.setattr(marina_app.llm, "call_llm", lambda _messages: generated_answer)

    response = marina_app.app.test_client().post(
        "/api/chat", json={"question": "Como funciona o forecast?"}
    )
    payload = response.get_json()

    assert response.status_code == 200
    assert payload["status"] == "no_context"
    assert payload["answer"] == marina_app.prompts.UNVERIFIED_SOURCES_MESSAGE
    assert payload["sources"] == []


def test_chat_returns_deterministic_best_model_metrics_without_llm(monkeypatch):
    monkeypatch.setattr(marina_app.retriever, "retrieve", lambda _question: [_chunk()])

    class MetricsProvider(EmptyMartProvider):
        def model_metrics(self):
            return [
                {
                    "model_name": "baseline_zero",
                    "model_family": "sanity_check",
                    "mae": 0.754,
                    "rmse": 2.416,
                    "mape": 100.0,
                    "wape": 100.0,
                    "is_best_model": True,
                },
                {
                    "model_name": "gradient_boosted_trees",
                    "model_family": "gradient_boosted_trees",
                    "mae": 1.333,
                    "rmse": 2.298,
                    "mape": 77.6758,
                    "wape": 176.792,
                    "is_best_model": False,
                },
            ]

    monkeypatch.setattr(marina_app, "mart_provider", MetricsProvider())
    monkeypatch.setattr(
        marina_app.llm,
        "call_llm",
        lambda _messages: pytest.fail("A resposta de métricas não deve chamar o LLM"),
    )

    response = marina_app.app.test_client().post(
        "/api/chat",
        json={
            "question": (
                "Qual modelo teve o menor WAPE no último run? Informe WAPE, MAE e RMSE."
            )
        },
    )

    payload = response.get_json()
    assert payload["status"] == "answered"
    assert "Baseline zero" in payload["answer"]
    assert "WAPE 100,000%" in payload["answer"]
    assert "MAE 0,754" in payload["answer"]
    assert "RMSE 2,416" in payload["answer"]
    assert "não comprova alta precisão" in payload["answer"]
    assert payload["sources"] == [
        "retail_analytics.default_marts.fct_forecast_model_metrics"
    ]


def test_chat_returns_deterministic_best_regression_comparison(monkeypatch):
    monkeypatch.setattr(marina_app.retriever, "retrieve", lambda _question: [_chunk()])

    class MetricsProvider(EmptyMartProvider):
        def model_metrics(self):
            return [
                {
                    "model_name": "baseline_zero",
                    "model_family": "sanity_check",
                    "wape": 100.0,
                    "is_best_model": True,
                },
                {
                    "model_name": "baseline_seasonal",
                    "model_family": "seasonal_naive",
                    "wape": 176.734,
                    "is_best_model": False,
                },
                {
                    "model_name": "gradient_boosted_trees",
                    "model_family": "gradient_boosted_trees",
                    "wape": 176.788,
                    "is_best_model": False,
                },
                {
                    "model_name": "random_forest",
                    "model_family": "random_forest",
                    "wape": 198.654,
                    "is_best_model": False,
                },
            ]

    monkeypatch.setattr(marina_app, "mart_provider", MetricsProvider())
    monkeypatch.setattr(
        marina_app.llm,
        "call_llm",
        lambda _messages: pytest.fail("A comparação não deve chamar o LLM"),
    )

    response = marina_app.app.test_client().post(
        "/api/chat",
        json={"question": "Qual foi o melhor modelo de regressão?"},
    )

    payload = response.get_json()
    assert payload["status"] == "answered"
    assert "Gradient Boosted Trees" in payload["answer"]
    assert "176,788%" in payload["answer"]
    assert "zero (100,000%)" in payload["answer"]
    assert "sazonal (176,734%)" in payload["answer"]
    assert "não foi promovido" in payload["answer"]


def test_chat_returns_deterministic_model_reliability_caveat(monkeypatch):
    monkeypatch.setattr(marina_app.retriever, "retrieve", lambda _question: [_chunk()])

    class MetricsProvider(EmptyMartProvider):
        def model_metrics(self):
            return [
                {
                    "model_name": "baseline_zero",
                    "wape": 100.0,
                    "is_best_model": True,
                }
            ]

    monkeypatch.setattr(marina_app, "mart_provider", MetricsProvider())
    monkeypatch.setattr(
        marina_app.llm,
        "call_llm",
        lambda _messages: pytest.fail("A ressalva não deve chamar o LLM"),
    )

    response = marina_app.app.test_client().post(
        "/api/chat",
        json={
            "question": (
                "Podemos afirmar que o melhor modelo avaliado é preciso e confiável?"
            )
        },
    )

    payload = response.get_json()
    assert payload["status"] == "answered"
    assert payload["answer"].startswith("Não.")
    assert "melhor somente entre os modelos avaliados" in payload["answer"]
    assert "não demonstra precisão ou confiabilidade" in payload["answer"]


def test_chat_routes_sku_replenishment_to_policy_and_answers_without_llm(monkeypatch):
    monkeypatch.setattr(
        marina_app.retriever,
        "retrieve",
        lambda _question: [_chunk("ml/demand_forecasting/README.md")],
    )

    class PolicyProvider(EmptyMartProvider):
        def forecast(self):
            return {
                "horizons": 3,
                "scope": "location_capacity_planning",
                "series": [{"period": "2026-09", "demand_qty": 5790.0}],
            }

        def forecast_publication(self):
            return {
                "series_count": 6054,
                "approved_series_count": 101,
                "review_required_series_count": 5953,
                "actionable_for_rankings": True,
            }

    monkeypatch.setattr(marina_app, "mart_provider", PolicyProvider())
    monkeypatch.setattr(
        marina_app.llm,
        "call_llm",
        lambda _messages: pytest.fail("A política SKU–loja não deve chamar o LLM"),
    )

    response = marina_app.app.test_client().post(
        "/api/chat",
        json={
            "question": (
                "A previsão por SKU–loja pode ser usada automaticamente para reposição?"
            )
        },
    )

    payload = response.get_json()
    assert payload["status"] == "answered"
    assert "não deve acionar reposição automática" in payload["answer"]
    assert "forecast agregado por localidade" in payload["answer"]
    assert payload["sources"] == [
        "retail_analytics.default_marts.fct_forecast_series_publication",
        "retail_analytics.default_marts.agg_location_demand_forecast",
    ]


@pytest.mark.parametrize(
    ("question", "source_path", "expected"),
    [
        (
            "O que causou a sazonalidade do produto 608?",
            "ml/demand_forecasting/README.md",
            "não permitem determinar sua causa",
        ),
        (
            "Como são calculadas as features utilizadas no forecast?",
            "ml/demand_forecasting/README.md",
            "year_number, quarter_number e month_number",
        ),
        (
            (
                "Por que preço atual, estoque atual e lead time atual não podem "
                "entrar no backtest histórico?"
            ),
            "ml/demand_forecasting/README.md",
            "vazamento temporal",
        ),
        (
            "Como podemos proteger as credenciais da aplicação?",
            "apps/marina/README.md",
            "OAuth do service principal",
        ),
    ],
)
def test_chat_returns_deterministic_documented_answers(
    monkeypatch, question, source_path, expected
):
    monkeypatch.setattr(
        marina_app.retriever,
        "retrieve",
        lambda _question: [_chunk(source_path)],
    )
    monkeypatch.setattr(
        marina_app.llm,
        "call_llm",
        lambda _messages: pytest.fail("A resposta documentada não deve chamar o LLM"),
    )

    response = marina_app.app.test_client().post(
        "/api/chat", json={"question": question}
    )

    payload = response.get_json()
    assert payload["status"] == "answered"
    assert expected in payload["answer"]
    assert payload["sources"]
    assert "DATABRICKS_" not in payload["answer"]


def test_chat_accepts_negated_seasonality_caveat_without_retry(monkeypatch):
    monkeypatch.setattr(marina_app.retriever, "retrieve", lambda _question: [_chunk()])

    class SeasonalityProvider(EmptyMartProvider):
        def seasonality(self):
            return {
                "product_id": 608,
                "confidence": "high",
                "series": [
                    {
                        "month_number": 12,
                        "seasonality_index": 2.97,
                        "seasonality_class": "peak",
                    }
                ],
            }

    monkeypatch.setattr(marina_app, "mart_provider", SeasonalityProvider())
    calls = []

    def fake_call_llm(messages):
        calls.append(messages)
        return (
            "O produto tem pico em dezembro. Confidence high significa pelo menos "
            "três observações em cada mês, mas não representa precisão, robustez "
            "ou confiabilidade estatística. [[SOURCES:S1]]"
        )

    monkeypatch.setattr(marina_app.llm, "call_llm", fake_call_llm)

    response = marina_app.app.test_client().post(
        "/api/chat",
        json={"question": "Como interpretar a sazonalidade do produto 608?"},
    )

    payload = response.get_json()
    assert payload["status"] == "answered"
    assert len(calls) == 1
    assert payload["sources"] == ["retail_analytics.default_marts.agg_product_seasonality"]


def test_chat_logs_first_and_final_validation_issues(monkeypatch, caplog):
    monkeypatch.setattr(marina_app.retriever, "retrieve", lambda _question: [_chunk()])
    generated_answer = "Resposta sem marcador."
    monkeypatch.setattr(
        marina_app.llm,
        "call_llm",
        lambda _messages: generated_answer,
    )

    with caplog.at_level(logging.INFO, logger=marina_app.app.logger.name):
        response = marina_app.app.test_client().post(
            "/api/chat",
            json={"question": "Como funciona o forecast?"},
        )

    assert response.get_json()["status"] == "no_context"
    assert "encaminhada para correção" in caplog.text
    assert "Correção da Marina rejeitada" in caplog.text
    assert "pelo menos uma fonte válida" in caplog.text
    assert generated_answer not in caplog.text


def test_chat_corrects_misleading_security_guidance_without_no_context(monkeypatch):
    monkeypatch.setattr(
        marina_app.retriever,
        "retrieve",
        lambda _question: [_chunk("apps/marina/README.md")],
    )
    answers = iter(
        [
            (
                "Evite tokens pessoais não versionados e use Personal Access "
                "Token. [[SOURCES:S1]]"
            ),
            (
                "Não exponha valores de credenciais em código, respostas ou logs; "
                "aplique permissões mínimas. [[SOURCES:S1]]"
            ),
        ]
    )
    calls = []

    def fake_call_llm(messages):
        calls.append(messages)
        return next(answers)

    monkeypatch.setattr(marina_app.llm, "call_llm", fake_call_llm)

    response = marina_app.app.test_client().post(
        "/api/chat",
        json={"question": "O que devo evitar para prevenir vazamento de dados?"},
    )

    payload = response.get_json()
    assert payload["status"] == "answered"
    assert len(calls) == 2
    assert payload["sources"] == ["apps/marina/README.md"]


def test_prepare_context_keeps_only_requested_location_ranking():
    context = {
        "locations": [
            {
                "location_name": "Loja A",
                "growth_abs_qty": 100,
                "growth_pct": 10,
                "growth_rank_abs": 1,
                "growth_rank_pct": 2,
            },
            {
                "location_name": "Loja B",
                "growth_abs_qty": 50,
                "growth_pct": 20,
                "growth_rank_abs": 2,
                "growth_rank_pct": 1,
            },
        ]
    }

    absolute = marina_app._prepare_operational_context(
        "Qual loja apresenta maior crescimento?", context
    )
    percentage = marina_app._prepare_operational_context(
        "Qual loja apresenta maior crescimento percentual?", context
    )

    assert absolute["locations"] == [
        {
            "location_name": "Loja A",
            "city": None,
            "state": None,
            "growth_abs_qty": 100,
            "growth_rank_abs": 1,
            "ranking_basis": "absolute",
        }
    ]
    assert percentage["locations"][0]["location_name"] == "Loja B"
    assert "growth_abs_qty" not in percentage["locations"][0]


def test_prepare_context_keeps_growth_periods_for_the_selected_leader():
    context = {
        "distribution_centers": [
            {
                "location_name": "Distribution Center 1",
                "growth_abs_qty": 5112,
                "growth_rank_abs": 1,
                "actual_start_month": "2026-06-01",
                "actual_end_month": "2026-08-01",
                "forecast_start_month": "2026-09-01",
                "forecast_end_month": "2026-11-01",
            }
        ]
    }

    prepared = marina_app._prepare_operational_context(
        "Qual CD teve o maior aumento previsto?", context
    )
    leader = prepared["distribution_centers"][0]

    assert leader["actual_start_month"] == "2026-06-01"
    assert leader["forecast_end_month"] == "2026-11-01"
    assert "demanda realizada" in leader["comparison_definition"]


@pytest.mark.parametrize(
    ("question", "context", "sources", "expected_answer"),
    [
        (
            "Qual loja teve o maior aumento previsto e entre quais meses?",
            {
                "locations": [
                    {
                        "location_name": "Store 1",
                        "growth_abs_qty": 2044,
                        "actual_start_month": "2026-06-01",
                        "actual_end_month": "2026-08-31",
                        "forecast_start_month": "2026-09-01",
                        "forecast_end_month": "2026-11-01",
                    }
                ]
            },
            ["retail_analytics.default_marts.agg_location_forecast_growth"],
            "Store 1 foi a loja com maior aumento previsto: 2.044 unidades. "
            "A comparação é a previsão de set-nov/26 versus a demanda realizada de "
            "jun-ago/26.",
        ),
        (
            "Qual CD teve o maior aumento previsto e entre quais meses?",
            {
                "distribution_centers": [
                    {
                        "location_name": "Distribution Center 1",
                        "growth_abs_qty": 5112,
                        "actual_start_month": "2026-06-01",
                        "actual_end_month": "2026-08-31",
                        "forecast_start_month": "2026-09-01",
                        "forecast_end_month": "2026-11-01",
                    }
                ]
            },
            ["retail_analytics.default_marts.agg_location_forecast_growth"],
            "Distribution Center 1 foi o CD com maior aumento previsto: 5.112 unidades. "
            "A comparação é a previsão de set-nov/26 versus a demanda realizada de "
            "jun-ago/26.",
        ),
        (
            "Qual região lidera a triagem de novo CD e o que significa o "
            "valor exibido?",
            {
                "recommendations": [
                    {
                        "macro_region": "Central-West",
                        "candidate_rank": 1,
                        "forecast_next_3m_demand_qty": 4005,
                    }
                ]
            },
            ["retail_analytics.default_marts.agg_distribution_center_recommendation"],
            "Central-West lidera a triagem. O valor exibido é 4.005 unidades "
            "previstas para os próximos três meses; não é crescimento, score nem "
            "decisão automática de abertura.",
        ),
        (
            "A Marina recomenda abrir um novo CD automaticamente?",
            {"recommendations": [{"candidate_rank": 1}]},
            ["retail_analytics.default_marts.agg_distribution_center_recommendation"],
            "Não. A triagem apenas prioriza uma análise; a abertura de um novo CD "
            "depende de capacidade, nível de serviço e custos, com decisão humana.",
        ),
    ],
)
def test_deterministic_answer_uses_current_growth_and_recommendation_values(
    question, context, sources, expected_answer
):
    result = marina_app._deterministic_answer(question, context, sources)

    assert result == (expected_answer, sources)


def test_deterministic_answer_uses_policy_count_and_short_cv_explanation():
    policy_result = marina_app._deterministic_answer(
        "Quantas séries SKU-loja exigem revisão no último run?",
        {"forecast_policy": {"review_required_series_count": 1845}},
        ["retail_analytics.default_marts.fct_forecast_series_publication"],
    )
    cv_result = marina_app._deterministic_answer(
        "O índice de oscilação prova que a previsão é precisa?",
        {"seasonality": {"cv": 0.908}, "model_metrics": [{"wape": 100}]},
        [
            "retail_analytics.default_marts.agg_product_seasonality",
            "retail_analytics.default_marts.fct_forecast_model_metrics",
        ],
    )

    assert policy_result == (
        "No último run, 1.845 séries SKU–loja exigem revisão.",
        ["retail_analytics.default_marts.fct_forecast_series_publication"],
    )
    assert cv_result == (
        "Não. O índice de oscilação (CV) mede a variação histórica da demanda; ele "
        "não mede a precisão da previsão. Essa precisão é avaliada no backtest por "
        "métricas como WAPE e MAPE.",
        [
            "retail_analytics.default_marts.agg_product_seasonality",
            "retail_analytics.default_marts.fct_forecast_model_metrics",
        ],
    )


def test_chat_answers_defensive_credential_question_without_calling_llm(monkeypatch):
    monkeypatch.setattr(
        marina_app.retriever,
        "retrieve",
        lambda _question: [_chunk("apps/marina/README.md")],
    )
    llm_call = Mock()
    monkeypatch.setattr(marina_app.llm, "call_llm", llm_call)

    response = marina_app.app.test_client().post(
        "/api/chat", json={"question": "Como as credenciais da Marina são protegidas?"}
    )

    assert response.get_json() == {
        "status": "answered",
        "answer": (
            "Nunca versione tokens ou credenciais. No runtime da aplicação, use "
            "OAuth do service principal e recursos valueFrom, aplique o menor "
            "privilégio e não exponha valores em código, respostas ou logs."
        ),
        "sources": ["apps/marina/README.md"],
        "question": "Como as credenciais da Marina são protegidas?",
    }
    llm_call.assert_not_called()


def test_chat_does_not_call_llm_without_context(monkeypatch):
    llm_call = Mock()
    monkeypatch.setattr(marina_app.retriever, "retrieve", lambda _question: [])
    monkeypatch.setattr(marina_app.llm, "call_llm", llm_call)

    response = marina_app.app.test_client().post(
        "/api/chat", json={"question": "Pergunta sem contexto"}
    )

    assert response.status_code == 200
    assert response.get_json()["status"] == "no_context"
    llm_call.assert_not_called()


def test_chat_degrades_when_retrieval_fails(monkeypatch):
    def fail_retrieval(_question):
        raise RuntimeError("índice indisponível")

    monkeypatch.setattr(marina_app.retriever, "retrieve", fail_retrieval)

    response = marina_app.app.test_client().post(
        "/api/chat", json={"question": "Qual loja mais cresceu?"}
    )

    assert response.status_code == 200
    assert response.get_json()["status"] == "unavailable"


def test_chat_degrades_without_sources_when_llm_fails(monkeypatch):
    def fail_llm(_messages):
        raise RuntimeError("endpoint indisponível")

    monkeypatch.setattr(marina_app.retriever, "retrieve", lambda _question: [_chunk()])
    monkeypatch.setattr(marina_app.llm, "call_llm", fail_llm)

    response = marina_app.app.test_client().post(
        "/api/chat", json={"question": "Qual loja mais cresceu?"}
    )

    payload = response.get_json()
    assert response.status_code == 200
    assert payload["status"] == "unavailable"
    assert payload["sources"] == []


def test_chat_validates_empty_and_oversized_questions():
    client = marina_app.app.test_client()

    empty_response = client.post("/api/chat", json={"question": "   "})
    oversized_response = client.post(
        "/api/chat", json={"question": "a" * (marina_app.MAX_QUESTION_LENGTH + 1)}
    )

    assert empty_response.status_code == 400
    assert oversized_response.status_code == 400


def test_chat_uses_location_growth_data_from_marts(monkeypatch):
    monkeypatch.setattr(marina_app.retriever, "retrieve", lambda _question: [])

    class GrowthMartProvider:
        def locations(self):
            return [{"location_name": "Distribution Center 1", "growth_abs_qty": 22.0}]

    monkeypatch.setattr(marina_app, "mart_provider", GrowthMartProvider())
    captured = {}

    def fake_call_llm(messages):
        captured["messages"] = messages
        return "Distribution Center 1 teve o maior crescimento. [[SOURCES:S1]]"

    monkeypatch.setattr(marina_app.llm, "call_llm", fake_call_llm)

    response = marina_app.app.test_client().post(
        "/api/chat", json={"question": "Qual loja mais cresceu?"}
    )

    assert response.status_code == 200
    assert response.get_json()["status"] == "answered"
    assert "Distribution Center 1" in captured["messages"][1]["content"]


def test_chat_includes_forecast_scope_for_network_questions(monkeypatch):
    monkeypatch.setattr(marina_app.retriever, "retrieve", lambda _question: [])

    class ForecastMartProvider:
        def forecast(self):
            return {
                "horizons": 3,
                "location_count": 6,
                "store_count": 3,
                "distribution_center_count": 3,
                "series": [{"period": "2026-09", "demand_qty": 5790}],
            }

        def forecast_publication(self):
            return {"actionable_for_rankings": True}

    monkeypatch.setattr(marina_app, "mart_provider", ForecastMartProvider())
    captured = {}

    def fake_call_llm(messages):
        captured["messages"] = messages
        return "É uma previsão consolidada da rede. [[SOURCES:S1]]"

    monkeypatch.setattr(marina_app.llm, "call_llm", fake_call_llm)

    response = marina_app.app.test_client().post(
        "/api/chat", json={"question": "A previsão é de uma loja ou da rede?"}
    )

    assert response.get_json()["status"] == "answered"
    prompt = captured["messages"][1]["content"]
    assert '"location_count":6' in prompt
    assert "3 lojas e 3 CDs" in prompt


def test_chat_falls_back_to_vector_context_without_mart_data(monkeypatch):
    monkeypatch.setattr(marina_app.retriever, "retrieve", lambda _question: [_chunk()])

    class EmptyMartProvider:
        def visuals(self):
            return {}

    monkeypatch.setattr(marina_app, "mart_provider", EmptyMartProvider())
    captured = {}

    def fake_call_llm(messages):
        captured["messages"] = messages
        return "Resposta usando só o contexto documental. [[SOURCES:S1]]"

    monkeypatch.setattr(marina_app.llm, "call_llm", fake_call_llm)

    response = marina_app.app.test_client().post(
        "/api/chat", json={"question": "Como funciona o cálculo de sazonalidade?"}
    )

    assert response.status_code == 200
    assert response.get_json()["status"] == "answered"
    assert "Dados operacionais" not in captured["messages"][1]["content"]


def test_chat_degrades_gracefully_when_marts_fail(monkeypatch):
    monkeypatch.setattr(marina_app.retriever, "retrieve", lambda _question: [_chunk()])

    class FailingMartProvider:
        def locations(self):
            raise RuntimeError("sem conexão com o warehouse")

    monkeypatch.setattr(marina_app, "mart_provider", FailingMartProvider())
    monkeypatch.setattr(
        marina_app.llm,
        "call_llm",
        lambda _messages: "Resposta documental. [[SOURCES:S1]]",
    )

    response = marina_app.app.test_client().post(
        "/api/chat", json={"question": "Qual loja mais cresceu?"}
    )

    assert response.status_code == 200
    assert response.get_json()["status"] == "answered"


def test_chat_sources_combine_docs_and_marts_without_duplicates(monkeypatch):
    monkeypatch.setattr(
        marina_app.retriever, "retrieve", lambda _question: [_chunk(), _chunk()]
    )

    calls = []

    class GrowthAndForecastMartProvider(EmptyMartProvider):
        def locations(self):
            calls.append("locations")
            return [{"location_name": "Distribution Center 1"}]

        def forecast(self):
            calls.append("forecast")
            return {"horizons": 3, "series": [{"period": "2026-10"}]}

    monkeypatch.setattr(marina_app, "mart_provider", GrowthAndForecastMartProvider())
    monkeypatch.setattr(
        marina_app.llm,
        "call_llm",
        lambda _messages: "Distribution Center 1 lidera. [[SOURCES:S1]]",
    )

    response = marina_app.app.test_client().post(
        "/api/chat", json={"question": "Qual loja mais cresceu?"}
    )

    sources = response.get_json()["sources"]
    assert len(sources) == len(set(sources))
    assert sources == ["retail_analytics.default_marts.agg_location_forecast_growth"]
    assert calls == ["locations"]


def test_chat_returns_no_context_when_chunks_and_marts_are_empty(monkeypatch):
    monkeypatch.setattr(marina_app.retriever, "retrieve", lambda _question: [])

    class EmptyMartProvider:
        def seasonality(self):
            return {
                "product_id": 608,
                "cv": None,
                "series": [],
            }

    monkeypatch.setattr(marina_app, "mart_provider", EmptyMartProvider())

    response = marina_app.app.test_client().post(
        "/api/chat", json={"question": "Como interpretar a sazonalidade?"}
    )

    assert response.status_code == 200
    assert response.get_json()["status"] == "no_context"


def test_chat_includes_model_metrics_and_seasonality_ranking(monkeypatch):
    monkeypatch.setattr(marina_app.retriever, "retrieve", lambda _question: [])

    class CompleteMartProvider(EmptyMartProvider):
        def model_metrics(self):
            return [
                {
                    "model_name": "lightgbm",
                    "wape": 0.18,
                    "is_best_model": True,
                }
            ]

        def seasonality_ranking(self):
            return [
                {
                    "product_variant_id": 608,
                    "cv": 0.908,
                    "peak_index": 2.97,
                }
            ]

    monkeypatch.setattr(marina_app, "mart_provider", CompleteMartProvider())
    captured = {}

    def fake_call_llm(messages):
        captured["messages"] = messages
        return "Os dados operacionais foram considerados. [[SOURCES:S1,S2]]"

    monkeypatch.setattr(marina_app.llm, "call_llm", fake_call_llm)

    response = marina_app.app.test_client().post(
        "/api/chat", json={"question": "Quais são os resultados atuais?"}
    )

    payload = response.get_json()
    prompt = captured["messages"][1]["content"]
    assert payload["status"] == "answered"
    assert "lightgbm" in prompt
    assert '"product_variant_id":608' in prompt
    assert "retail_analytics.default_marts.fct_forecast_model_metrics" in payload["sources"]
    assert "retail_analytics.default_marts.agg_product_seasonality" in payload["sources"]


def test_mart_sources_use_configured_catalog_and_schema_without_duplicates():
    class ConfiguredProvider:
        catalog = "custom_catalog"
        schema = "custom_marts"

    context = {
        "seasonality": {"series": [{"product_variant_id": 608}]},
        "seasonality_ranking": [{"product_variant_id": 608}],
    }

    assert marina_app._mart_sources(context, ConfiguredProvider()) == [
        "custom_catalog.custom_marts.agg_product_seasonality"
    ]


def test_operational_loaders_degrade_independently():
    class PartiallyAvailableProvider(EmptyMartProvider):
        def locations(self):
            return [{"location_name": "Distribution Center 1"}]

        def model_metrics(self):
            raise RuntimeError("tabela de métricas indisponível")

        def seasonality_ranking(self):
            return [{"product_variant_id": 608, "cv": 0.908}]

    context = marina_app._load_operational_context(
        PartiallyAvailableProvider(),
        ("locations", "model_metrics", "seasonality_ranking"),
    )

    assert context["locations"][0]["location_name"] == "Distribution Center 1"
    assert "model_metrics" not in context
    assert context["seasonality_ranking"][0]["product_variant_id"] == 608


def test_provider_queries_latest_model_metrics_and_seasonality_ranking(monkeypatch):
    provider = marina_app.DatabricksMartProvider()
    queries = []

    def fake_query(query):
        queries.append(query)
        return []

    monkeypatch.setattr(provider, "_query", fake_query)

    assert provider.model_metrics() == []
    assert provider.seasonality_ranking() == []

    assert "fct_forecast_model_metrics" in queries[0]
    assert "m.is_best_model" in queries[0]
    assert "agg_product_seasonality" in queries[1]
    assert "min_observations_per_month" in queries[1]
    assert "confidence" in queries[1]
    assert "LIMIT 10" in queries[1]


def test_provider_separates_store_and_distribution_center_rankings(monkeypatch):
    provider = marina_app.DatabricksMartProvider()
    queries = []
    monkeypatch.setattr(provider, "_query", lambda query: queries.append(query) or [])

    assert provider.locations() == []
    assert provider.distribution_centers() == []

    store_query, distribution_center_query = queries
    for query in queries:
        assert "growth_rank_abs" in query
        assert "growth_rank_pct" in query
        assert "actual_start_month" in query
        assert "actual_end_date AS actual_end_month" in query
        assert "forecast_start_month" in query
        assert "forecast_end_month" in query
        assert "growth_rank_abs <= 5 OR growth_rank_pct <= 5" in query
        assert "ORDER BY growth_rank_abs" in query
        assert "agg_location_demand_forecast" in query
        assert "COUNT(DISTINCT horizon_month) = 3" in query
        assert "MAX(created_at) AS run_created_at" in query
        assert "ORDER BY run_created_at DESC, run_id DESC" in query
    assert "comparison_type = 'store'" in store_query
    assert "comparison_type = 'distribution_center'" in distribution_center_query


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        (
            "Qual loja apresenta o maior crescimento absoluto?",
            ("locations", "forecast_policy"),
        ),
        (
            "Como interpretar a sazonalidade do Outboard Motor 2640?",
            ("seasonality",),
        ),
        ("Quais produtos são mais sazonais?", ("seasonality_ranking",)),
        ("Como os modelos são avaliados?", ("model_metrics",)),
        (
            "Devemos abrir um novo CD?",
            ("recommendations", "forecast_policy"),
        ),
        ("Como funciona a previsão de demanda?", ("forecast", "forecast_policy")),
        (
            "Qual é a previsão para o próximo mês?",
            ("forecast", "forecast_policy"),
        ),
        (
            "A previsão por SKU–loja pode ser usada automaticamente para reposição?",
            ("forecast", "forecast_policy"),
        ),
        (
            "A previsão de setembro é de uma loja ou da rede toda?",
            ("forecast", "forecast_policy"),
        ),
        ("Quantas lojas e CDs entram nesta previsão?", ("forecast", "forecast_policy")),
        ("O valor de novembro é acumulado?", ("forecast", "forecast_policy")),
        (
            "Essas unidades previstas representam faturamento?",
            ("forecast", "forecast_policy"),
        ),
        ("O produto em destaque foi escolhido manualmente?", ("seasonality",)),
        (
            "O que significam as cores do gráfico de sazonalidade?",
            ("seasonality",),
        ),
        (
            "O índice de oscilação prova que a previsão é precisa?",
            ("forecast", "forecast_policy", "model_metrics", "seasonality"),
        ),
        (
            "Qual CD tem o maior aumento previsto?",
            ("distribution_centers", "forecast_policy"),
        ),
        (
            "Quantas séries SKU-loja exigem revisão no último run?",
            ("forecast_policy",),
        ),
        (
            "Qual região lidera a triagem de novo CD?",
            ("recommendations", "forecast_policy"),
        ),
        ("Qual loja tem maior demanda prevista?", ()),
        ("Como está organizada a arquitetura?", ()),
        ("Quais são os indicadores atuais?", marina_app.ALL_OPERATIONAL_CONTEXT_KEYS),
    ],
)
def test_operational_context_keys_selects_relevant_marts(question, expected):
    assert marina_app._operational_context_keys(question) == expected


@pytest.mark.parametrize(
    ("minimum", "expected"),
    [(None, None), (1, "low"), (2, "medium"), (3, "high"), (5, "high")],
)
def test_seasonality_confidence_uses_minimum_monthly_observations(minimum, expected):
    assert (
        marina_app.DatabricksMartProvider._seasonality_confidence(minimum) == expected
    )


def test_provider_seasonality_includes_classes_observations_and_confidence(monkeypatch):
    monkeypatch.delenv("MARINA_SEASONALITY_PRODUCT_ID", raising=False)
    provider = marina_app.DatabricksMartProvider()
    queries = []
    rows = [
        {
            "product_variant_id": 608,
            "sku": "SKU-462189",
            "product_name": "Outboard Motor 2640",
            "month_number": 1,
            "seasonality_index": 2.259,
            "seasonality_class": "peak",
            "observations": 4,
            "cv": 0.908,
        },
        {
            "product_variant_id": 608,
            "sku": "SKU-462189",
            "product_name": "Outboard Motor 2640",
            "month_number": 2,
            "seasonality_index": 1.397,
            "seasonality_class": "peak",
            "observations": 3,
            "cv": 0.908,
        },
    ]

    def fake_query(query):
        queries.append(query)
        return rows

    monkeypatch.setattr(provider, "_query", fake_query)

    seasonality = provider.seasonality()

    assert seasonality["product_name"] == "Outboard Motor 2640"
    assert seasonality["min_observations_per_month"] == 3
    assert seasonality["confidence"] == "high"
    assert seasonality["series"][0]["seasonality_class"] == "peak"
    assert seasonality["series"][0]["observations"] == 4
    assert seasonality["selection_method"] == "highest_cv_minimum_monthly_observations"
    assert "Destaque automático" in seasonality["selection_description"]
    assert "seasonality_class" in queries[0]
    assert "observations" in queries[0]
    assert "HAVING MIN(observations) >= 3" in queries[0]
    assert "WHERE product_variant_id =" not in queries[0]


def test_provider_seasonality_allows_configured_product_override(monkeypatch):
    monkeypatch.setenv("MARINA_SEASONALITY_PRODUCT_ID", "777")
    provider = marina_app.DatabricksMartProvider()
    queries = []
    monkeypatch.setattr(provider, "_query", lambda query: queries.append(query) or [])

    seasonality = provider.seasonality()

    assert seasonality["product_id"] is None
    assert seasonality["selection_method"] == "configured_product"
    assert "SELECT 777 AS product_variant_id" in queries[0]


def test_provider_visuals_preserves_dashboard_contract(monkeypatch):
    provider = marina_app.DatabricksMartProvider()
    monkeypatch.setattr(
        provider,
        "forecast",
        lambda: {"horizons": 1, "series": [{"period": "2026-10"}]},
    )
    monkeypatch.setattr(
        provider,
        "seasonality",
        lambda: {"product_id": 608, "series": [{"month_number": 1}]},
    )
    monkeypatch.setattr(
        provider,
        "locations",
        lambda: [{"location_name": "Distribution Center 1"}],
    )
    monkeypatch.setattr(
        provider,
        "distribution_centers",
        lambda: [{"location_name": "Distribution Center 1"}],
    )
    monkeypatch.setattr(
        provider,
        "recommendations",
        lambda: [
            {"macro_region": "Central-West", "candidate_rank": 1},
            {"macro_region": "Southeast", "candidate_rank": 2},
        ],
    )
    monkeypatch.setattr(
        provider,
        "forecast_publication",
        lambda: {"actionable_for_rankings": True},
    )

    payload = provider.visuals()

    assert payload["status"] == "ready"
    assert payload["forecast"]["horizons"] == 1
    assert payload["seasonality"]["product_id"] == 608
    assert payload["locations"][0]["location_name"] == "Distribution Center 1"
    assert payload["distribution_centers"][0]["location_name"] == "Distribution Center 1"
    assert payload["forecast_policy"]["actionable_for_rankings"] is True
    assert payload["top_candidate"]["macro_region"] == "Central-West"


def test_provider_recommendations_include_current_growth_and_forecast_run(monkeypatch):
    provider = marina_app.DatabricksMartProvider()
    queries = []

    def fake_query(query):
        queries.append(query)
        return []

    monkeypatch.setattr(provider, "_query", fake_query)

    assert provider.recommendations() == []

    query = queries[0]
    assert "actual_last_3m_demand_qty" in query
    assert "forecast_next_3m_demand_qty" in query
    assert "growth_abs_qty" in query
    assert "growth_pct" in query
    assert "source_forecast_run_id" in query
    assert "agg_location_demand_forecast" in query
    assert "COUNT(DISTINCT horizon_month) = 3" in query
    assert "MAX(created_at) AS run_created_at" in query
    assert "ORDER BY run_created_at DESC, run_id DESC" in query


def test_provider_forecast_uses_aggregate_and_keeps_series_status(monkeypatch):
    provider = marina_app.DatabricksMartProvider()
    queries = []

    def fake_query(query):
        queries.append(query)
        return []

    monkeypatch.setattr(provider, "_query", fake_query)

    assert provider.forecast() == {
        "horizons": 0,
        "scope": "location_capacity_planning",
        "location_count": 0,
        "store_count": 0,
        "distribution_center_count": 0,
        "series": [],
    }
    assert provider.forecast_publication() == {}

    assert "agg_location_demand_forecast" in queries[0]
    assert "COUNT(DISTINCT f.location_id) AS location_count" in queries[0]
    assert "location_type = 'store'" in queries[0]
    assert "location_type = 'warehouse'" in queries[0]
    assert "MAX(created_at) AS run_created_at" in queries[0]
    assert "ORDER BY run_created_at DESC, run_id DESC" in queries[0]
    assert "fct_forecast_series_publication" in queries[1]
    assert "agg_location_demand_forecast" in queries[1]


def test_provider_builds_warehouse_http_path_from_app_resource(monkeypatch):
    monkeypatch.setenv("DATABRICKS_HOST", "https://workspace.example.com")
    monkeypatch.setenv("DATABRICKS_WAREHOUSE_ID", "warehouse-123")
    monkeypatch.setenv("DATABRICKS_CLIENT_ID", "client-123")
    monkeypatch.setenv("DATABRICKS_CLIENT_SECRET", "secret-123")
    monkeypatch.delenv("DATABRICKS_HTTP_PATH", raising=False)

    provider = marina_app.DatabricksMartProvider()

    assert provider.configured
    assert provider._server_hostname() == "workspace.example.com"
    assert provider._http_path() == "/sql/1.0/warehouses/warehouse-123"
