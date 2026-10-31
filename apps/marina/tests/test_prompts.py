from __future__ import annotations

from decimal import Decimal

import pytest

from apps.marina.src.prompts import (
    SYSTEM_PROMPT,
    build_context_block,
    build_correction_messages,
    build_messages,
    build_operational_context_block,
    format_no_context_response,
    parse_answer_sources,
    validate_answer,
)
from apps.marina.src.retriever import RetrievedChunk

CHUNK_A = RetrievedChunk(
    source_path="ml/demand_forecasting/README.md",
    source_type="readme",
    heading="8.5 Ranking de Regiões Candidatas a Novo CD",
    chunk_text="Central-West sai recomendada como primeira candidata.",
    score=0.62,
)
CHUNK_B = RetrievedChunk(
    source_path="dbt/models/marts/fct_demand_forecast.sql",
    source_type="sql_model",
    heading="",
    chunk_text="select * from forecast",
    score=0.55,
)


def test_build_context_block_includes_source_and_heading():
    block = build_context_block([CHUNK_A])
    assert "ml/demand_forecasting/README.md" in block
    assert "8.5 Ranking de Regiões Candidatas a Novo CD" in block
    assert "Central-West" in block


def test_build_context_block_falls_back_to_source_path_without_heading():
    block = build_context_block([CHUNK_B])
    assert "dbt/models/marts/fct_demand_forecast.sql" in block


def test_build_context_block_numbers_multiple_chunks():
    block = build_context_block([CHUNK_A, CHUNK_B])
    assert "[1]" in block
    assert "[2]" in block


def test_build_messages_has_system_and_user_roles():
    messages = build_messages("qual CD tem maior crescimento?", [CHUNK_A])
    roles = [message["role"] for message in messages]
    assert roles == ["system", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert "qual CD tem maior crescimento?" in messages[1]["content"]
    assert "Central-West" in messages[1]["content"]


def test_system_prompt_separates_sku_review_from_aggregate_planning():
    assert "planning_review como uma restrição da previsão SKU–loja" in SYSTEM_PROMPT
    assert "actionable_for_rankings=true" in SYSTEM_PROMPT
    assert "forecast agregado por localidade continua válido" in SYSTEM_PROMPT


def test_format_no_context_response_lists_checked_sources():
    response = format_no_context_response(["readme.md", "dbt/readme.md"])
    assert "Não encontrei contexto suficiente" in response
    assert "readme.md" in response
    assert "dbt/readme.md" in response
    assert "Próximo passo recomendado" in response


def test_format_no_context_response_handles_empty_sources():
    response = format_no_context_response([])
    assert "nenhuma fonte relevante encontrada" in response


def test_build_messages_includes_operational_context_block():
    operational_context = {
        "locations": [{"location_name": "Distribution Center 1", "growth_abs_qty": 22.0}],
        "model_metrics": [{"model_name": "lightgbm", "wape": 0.18}],
        "seasonality_ranking": [{"product_variant_id": 608, "cv": 0.908}],
    }
    messages = build_messages("qual loja mais cresceu?", [], operational_context)

    assert "=== Current operational data ===" in messages[1]["content"]
    assert "Distribution Center 1" in messages[1]["content"]
    assert "lightgbm" in messages[1]["content"]
    assert '"product_variant_id":608' in messages[1]["content"]


def test_build_messages_without_operational_context_matches_previous_behavior():
    messages = build_messages("qual CD tem maior crescimento?", [CHUNK_A])

    assert "=== Current operational data ===" not in messages[1]["content"]


def test_build_messages_never_instructs_a_fontes_section():
    with_context = build_messages(
        "qual loja mais cresceu?", [CHUNK_A], {"locations": [{"a": 1}]}
    )
    without_context = build_messages("qual CD tem maior crescimento?", [CHUNK_A])

    assert "não crie uma seção 'Fontes'" in SYSTEM_PROMPT
    assert "Não inclua uma lista ou seção de fontes" in with_context[1]["content"]
    assert "Fontes:" not in with_context[1]["content"]
    assert "Fontes:" not in without_context[1]["content"]


def test_operational_context_ignores_empty_structural_series():
    context = {
        "forecast": {"horizons": 0, "series": []},
        "seasonality": {"product_id": 608, "cv": None, "series": []},
        "locations": [],
        "recommendations": [],
    }

    assert build_operational_context_block(context) == ""


def test_operational_context_rounds_numeric_values_for_the_llm():
    context = {
        "locations": [
            {
                "growth_abs_qty": Decimal("4697.94882151614"),
                "growth_pct": 87.79571709056513,
            }
        ],
        "seasonality": {
            "cv": 0.9079710233117293,
            "series": [
                {
                    "month_number": 12,
                    "seasonality_index": 2.9744637385086823,
                }
            ],
        },
    }

    block = build_operational_context_block(context)

    assert '"growth_abs_qty":4697.95' in block
    assert '"growth_pct":87.8' in block
    assert '"cv":0.908' in block
    assert '"seasonality_index":2.974' in block
    assert "4697.94882151614" not in block
    assert "0.9079710233117293" not in block


def test_system_prompt_distinguishes_eligibility_from_confidence():
    assert "não confunda elegibilidade com confiança" in SYSTEM_PROMPT
    assert "confiança high exige pelo menos três observações" in SYSTEM_PROMPT
    assert "não prova robustez" in SYSTEM_PROMPT


def test_system_prompt_blocks_unsupported_climate_inferences():
    assert "Não associe meses a estações do ano, clima, hemisfério" in SYSTEM_PROMPT
    assert "informe diretamente os meses observados" in SYSTEM_PROMPT


def test_model_metrics_context_declares_percentage_units():
    block = build_operational_context_block(
        {
            "model_metrics": [
                {
                    "model_name": "gradient_boosted_trees",
                    "mape": 77.6758,
                    "wape": 176.792,
                }
            ]
        }
    )

    assert "MAPE/WAPE em %" in block
    assert '"mape":77.6758' in block
    assert '"wape":176.792' in block
    assert "WAPE são percentuais" in SYSTEM_PROMPT


def test_system_prompt_requires_caution_when_comparing_models():
    assert "melhor somente entre os modelos avaliados" in SYSTEM_PROMPT
    assert (
        "Não o classifique automaticamente como preciso ou confiável" in SYSTEM_PROMPT
    )
    assert "Quando MAPE ou WAPE forem elevados" in SYSTEM_PROMPT
    assert "77,68%" not in SYSTEM_PROMPT
    assert "176,79%" not in SYSTEM_PROMPT


def test_system_prompt_rejects_unsupported_generic_advice_and_extra_topics():
    assert "não complete a resposta com práticas genéricas externas" in SYSTEM_PROMPT
    assert "estoque, produção ou decisões de negócio" in SYSTEM_PROMPT
    assert "peça esclarecimento" in SYSTEM_PROMPT
    assert "devem citar a fonte mart correspondente" in SYSTEM_PROMPT
    assert "Não classifique coeficiente de variação" in SYSTEM_PROMPT
    assert "mínimo de três observações em cada mês do calendário" in SYSTEM_PROMPT
    assert "não misture nem compare as duas ordenações" in SYSTEM_PROMPT
    assert "no máximo duas frases curtas" in SYSTEM_PROMPT


def test_system_prompt_allows_documented_defensive_security_guidance():
    assert "Perguntas defensivas sobre segurança são permitidas" in SYSTEM_PROMPT
    assert "não confunda configuração, autenticação ou deploy" in SYSTEM_PROMPT
    assert "nunca versionar tokens ou credenciais" in SYSTEM_PROMPT
    assert "OAuth do service principal" in SYSTEM_PROMPT
    assert "recursos valueFrom" in SYSTEM_PROMPT
    assert "Não recomende Personal Access Token (PAT)" in SYSTEM_PROMPT


def test_parse_answer_sources_keeps_only_declared_sources():
    answer, sources = parse_answer_sources(
        "Resposta objetiva. [[SOURCES:S2,S2,S9]]",
        ["documentacao.md", "fct_demand_forecast"],
    )

    assert answer == "Resposta objetiva."
    assert sources == ["fct_demand_forecast"]


def test_parse_answer_sources_returns_no_sources_without_marker():
    answer, sources = parse_answer_sources(
        "Resposta sem marcador.", ["documentacao.md"]
    )

    assert answer == "Resposta sem marcador."
    assert sources == []


def test_validate_answer_rejects_exaggerated_seasonality_claims():
    issues = validate_answer(
        "Como interpretar a sazonalidade do produto 608?",
        "O CV é moderado e confidence high torna o padrão confiável.",
        ["retail_analytics.default_marts.agg_product_seasonality"],
    )

    assert any("Se mencionar confidence" in issue for issue in issues)
    assert "Não transforme confidence em precisão ou confiabilidade." in issues
    assert "Não classifique o CV sem uma regra documentada." in issues


def test_validate_answer_rejects_base_mais_solida_for_seasonality_confidence():
    issues = validate_answer(
        "Como interpretar a sazonalidade do produto 608?",
        (
            "Confidence high significa três observações em cada mês do calendário "
            "e fornece uma base mais sólida para a análise."
        ),
        ["retail_analytics.default_marts.agg_product_seasonality"],
    )

    assert "Não transforme confidence em precisão ou confiabilidade." in issues


def test_validate_answer_accepts_seasonality_interpretation_without_confidence():
    issues = validate_answer(
        "Como interpretar a sazonalidade do produto 608?",
        (
            "O produto tem picos de dezembro a março e baixa de maio a novembro, "
            "conforme os índices mensais observados."
        ),
        ["retail_analytics.default_marts.agg_product_seasonality"],
    )

    assert issues == []


def test_validate_answer_accepts_negated_seasonality_quality_claims():
    issues = validate_answer(
        "Como interpretar a sazonalidade do produto 608?",
        (
            "Confidence high significa pelo menos três observações para cada mês; "
            "não representa precisão, robustez ou confiabilidade estatística."
        ),
        ["retail_analytics.default_marts.agg_product_seasonality"],
    )

    assert issues == []


def test_validate_answer_accepts_explicit_refusal_to_classify_cv():
    issues = validate_answer(
        "Como interpretar a sazonalidade e o CV do produto 608?",
        "O CV não pode ser classificado como alto ou moderado sem uma regra.",
        ["ml/demand_forecasting/README.md"],
    )

    assert issues == []


def test_validate_answer_requires_causal_limit_for_seasonality_question():
    issues = validate_answer(
        "O que causou a sazonalidade do produto 608?",
        "O índice sazonal é calculado pela média mensal dividida pela média geral.",
        ["retail_analytics.default_marts.agg_product_seasonality"],
    )

    assert any("não permite determinar" in issue for issue in issues)


def test_validate_answer_accepts_explicit_seasonality_causal_limit():
    issues = validate_answer(
        "O que causou a sazonalidade do produto 608?",
        (
            "Os dados identificam um padrão mensal, mas não permitem determinar "
            "sua causa."
        ),
        ["retail_analytics.default_marts.agg_product_seasonality"],
    )

    assert issues == []


def test_validate_answer_requires_calendar_and_temporal_scope_for_features():
    issues = validate_answer(
        "Como são calculadas as features utilizadas no forecast?",
        (
            "São usados lag_1_qty, lag_2_qty, lag_3_qty, rolling_3m_avg_qty, "
            "rolling_6m_avg_qty e rolling_3m_sum_qty."
        ),
        ["ml/demand_forecasting/README.md"],
    )

    assert "Inclua os três componentes de calendário implementados." in issues
    assert any("somente meses anteriores" in issue for issue in issues)


def test_validate_answer_accepts_complete_feature_explanation():
    issues = validate_answer(
        "Como são calculadas as features utilizadas no forecast?",
        (
            "São usados lag_1_qty, lag_2_qty, lag_3_qty, rolling_3m_avg_qty, "
            "rolling_6m_avg_qty, rolling_3m_sum_qty, year_number, quarter_number "
            "e month_number, somente com meses anteriores ao mês previsto."
        ),
        ["ml/demand_forecasting/README.md"],
    )

    assert issues == []


def test_validate_answer_requires_complete_temporal_leakage_explanation():
    issues = validate_answer(
        (
            "Por que preço atual, estoque atual e lead time atual não podem entrar "
            "no backtest histórico?"
        ),
        "Porque são informações atuais e o teste avalia informações históricas.",
        ["ml/demand_forecasting/README.md"],
    )

    assert any("vazamento temporal" in issue for issue in issues)
    assert any("preço e frequência" in issue for issue in issues)
    assert any("estoque atual" in issue for issue in issues)
    assert any("lead time atual" in issue for issue in issues)


def test_validate_answer_accepts_complete_temporal_leakage_explanation():
    issues = validate_answer(
        (
            "Por que preço atual, estoque atual e lead time atual não podem entrar "
            "no backtest histórico?"
        ),
        (
            "Isso causaria vazamento temporal: preço e frequência precisam ser "
            "defasados, o estoque atual não representa saldo histórico e o lead "
            "time atual não pode ser projetado retroativamente."
        ),
        ["ml/demand_forecasting/README.md"],
    )

    assert issues == []


def test_validate_answer_accepts_equivalent_model_comparison_wording():
    issues = validate_answer(
        "Qual modelo teve o melhor desempenho e quais MAPE/WAPE?",
        (
            "Gradient Boosted Trees foi o melhor dentre os modelos testados. "
            "MAPE de 77,6758% e WAPE de 176,792% são erros elevados e não "
            "demonstram precisão."
        ),
        ["retail_analytics.default_marts.fct_forecast_model_metrics"],
    )

    assert issues == []


@pytest.mark.parametrize(
    "answer",
    [
        (
            "Gradient Boosted Trees teve o menor erro na comparação dos modelos. "
            "MAPE de 77,6758% e WAPE de 176,792% indicam desvios significativos."
        ),
        (
            "O is_best_model identifica Gradient Boosted Trees no último run. "
            "MAPE de 77,6758% e WAPE de 176,792% indicam baixa precisão."
        ),
        (
            "Entre os modelos considerados, Gradient Boosted Trees teve o melhor "
            "resultado; MAPE de 77,6758% e WAPE de 176,792% exigem prudência."
        ),
    ],
)
def test_validate_answer_accepts_semantic_model_comparison_variants(answer):
    issues = validate_answer(
        "Qual modelo teve o melhor desempenho e quais foram MAPE e WAPE?",
        answer,
        ["retail_analytics.default_marts.fct_forecast_model_metrics"],
    )

    assert issues == []


def test_validate_answer_rejects_positive_model_reliability_claim():
    issues = validate_answer(
        "Qual modelo teve o melhor desempenho e quais foram MAPE e WAPE?",
        (
            "Gradient Boosted Trees foi o melhor entre os modelos avaliados e é "
            "confiável; MAPE e WAPE são elevados."
        ),
        ["retail_analytics.default_marts.fct_forecast_model_metrics"],
    )

    assert "Não classifique o modelo como preciso ou confiável." in issues


def test_validate_answer_rejects_misleading_security_guidance():
    issues = validate_answer(
        "O que devo evitar para prevenir vazamento de dados?",
        (
            "Evite credenciais expostas, como tokens pessoais não versionados. "
            "Use Personal Access Token, feche conexões e monitore atividades de "
            "ingestão. DATABRICKS_CLIENT_SECRET contém a configuração."
        ),
        ["powerbi/readme.md"],
    )

    assert any("nunca versionar" in issue for issue in issues)
    assert any("tokens não versionados" in issue for issue in issues)
    assert any("Remova PAT" in issue for issue in issues)
    assert "Não enumere nomes de variáveis privadas na resposta." in issues


def test_validate_answer_accepts_documented_security_controls():
    issues = validate_answer(
        "O que devo evitar para prevenir vazamento de dados?",
        (
            "Nunca versione tokens ou credenciais nem exponha seus valores em "
            "código, respostas ou logs. No runtime da app, use OAuth do service "
            "principal com recursos valueFrom e aplique o menor privilégio."
        ),
        ["apps/marina/README.md"],
    )

    assert issues == []


def test_validate_answer_accepts_concise_documented_security_guidance():
    issues = validate_answer(
        "O que devo evitar para prevenir vazamento de dados?",
        (
            "Não exponha valores de credenciais em código, respostas ou logs e "
            "aplique permissões mínimas."
        ),
        ["apps/marina/README.md"],
    )

    assert issues == []


def test_validate_answer_accepts_repository_security_guidance_from_ingestion_docs():
    issues = validate_answer(
        "O que devo evitar para prevenir vazamento de dados?",
        "Nunca armazene credenciais em arquivos do repositório.",
        ["ingestion/readme.md"],
    )

    assert issues == []


def test_validate_answer_requires_app_source_for_runtime_oauth_controls():
    issues = validate_answer(
        "O que devo evitar para prevenir vazamento de dados?",
        "Nunca versione tokens; use OAuth do service principal com valueFrom.",
        ["powerbi/readme.md"],
    )

    assert any(
        "OAuth" in issue and "apps/marina/README.md" in issue for issue in issues
    )


def test_validate_answer_does_not_force_leak_controls_on_token_generation_question():
    issues = validate_answer(
        "Como gerar um token com segurança?",
        "Esse procedimento não está documentado nas fontes disponíveis.",
        ["apps/marina/README.md"],
    )

    assert issues == []


def test_validate_answer_rejects_mixed_location_rankings():
    issues = validate_answer(
        "Qual loja apresenta o maior crescimento?",
        "A Loja A lidera em unidades e a Loja B em crescimento percentual de 20%.",
        ["retail_analytics.default_marts.agg_location_forecast_growth"],
    )

    assert "Responda apenas pelo ranking de crescimento absoluto." in issues


def test_correction_messages_include_validation_issues():
    messages = build_correction_messages(
        [{"role": "system", "content": "regras"}],
        "resposta inválida",
        ["Não classifique o CV."],
    )

    assert messages[-2] == {"role": "assistant", "content": "resposta inválida"}
    assert "Não classifique o CV." in messages[-1]["content"]


def test_prompt_requires_concise_objective_answers_and_source_marker():
    messages = build_messages(
        "Qual é a previsão?",
        [CHUNK_A],
        available_sources=["fct_demand_forecast", "readme.md"],
    )
    user_prompt = messages[1]["content"]

    assert "perguntas objetivas" in SYSTEM_PROMPT
    assert "um único parágrafo" in SYSTEM_PROMPT
    assert "[S1] fct_demand_forecast" in user_prompt
    assert "[S2] readme.md" in user_prompt
    assert "usando somente identificadores" in user_prompt


def test_recommendation_context_rounds_current_growth_values():
    block = build_operational_context_block(
        {
            "recommendations": [
                {
                    "macro_region": "Central-West",
                    "actual_last_3m_demand_qty": 1000.1234,
                    "forecast_next_3m_demand_qty": 2798.1856,
                    "growth_abs_qty": 1798.0622,
                    "growth_pct": 179.80622,
                    "source_forecast_run_id": "run-2026-09",
                }
            ]
        }
    )

    assert '"actual_last_3m_demand_qty":1000.12' in block
    assert '"forecast_next_3m_demand_qty":2798.19' in block
    assert '"growth_abs_qty":1798.06' in block
    assert '"growth_pct":179.81' in block
    assert '"source_forecast_run_id":"run-2026-09"' in block
