"""Deterministic guardrails for questions sent to Marina."""

from __future__ import annotations

import re
import unicodedata

SAFE_REFUSAL_MESSAGE = (
    "I cannot help bypass safety instructions or provide, locate, "
    "or enumerate sensitive data, credentials, private configuration, or internal prompts. "
    "I can answer authorized technical questions about the results "
    "and public documentation for the Retail Analytics project."
)

_OVERRIDE_PATTERNS = (
    re.compile(
        r"\b(?:ignore|forget|disregard|override|bypass)\b"
        r"[\s\S]{0,100}\b(?:instructions?|prompt|system|policy|policies|rules?)\b"
    ),
    re.compile(
        r"\b(?:ignore|esqueca|desconsidere|contorne|burle)\b"
        r"[\s\S]{0,100}\b(?:instrucoes|prompt|sistema|politicas?|regras?)\b"
    ),
    re.compile(
        r"\b(?:reveal|repeat|show|print)\b[\s\S]{0,100}\b(?:system prompt|"
        r"(?:your|internal|previous) (?:instructions?|rules?))\b"
    ),
    re.compile(
        r"\b(?:revele|repita|mostre)\b[\s\S]{0,100}\b(?:prompt (?:do|de) sistema|"
        r"prompt (?:interno|oculto)|"
        r"(?:suas?|internas?|anteriores?|ocultas?) (?:instrucoes|regras)|"
        r"(?:instrucoes|regras) (?:internas?|anteriores?|ocultas?|do sistema))\b"
    ),
)

_SENSITIVE_TERMS = re.compile(
    r"(?:\b(?:dados? sensiveis?|dados? confidenciais?|dados? pessoais?|"
    r"credenciais?|senhas?|passwords?|"
    r"segredos?|secrets?|tokens?|client secret|api keys?|chaves? de api|"
    r"variaveis? de ambiente|"
    r"cpfs?|cnpjs?|rgs?|cart(?:ao|oes)|dados? bancarios?|contas? bancarias?|"
    r"chaves? pix)\b|\.env\b)"
)
_OBFUSCATED_SENSITIVE_TERMS = re.compile(
    r"(?:\bt[\W_]*o[\W_]*k[\W_]*e[\W_]*n(?:s)?\b|"
    r"\bs[\W_]*e[\W_]*n[\W_]*h[\W_]*a(?:s)?\b)"
)
_CUSTOMER_PERSONAL_DATA_TERMS = re.compile(
    r"(?:\b(?:numero|id|identificador|telefone|celular|e-?mail|endereco|cep|"
    r"documento|number|phone|address|zip|card|document)s?\b.{0,60}"
    r"\b(?:cliente|consumidor|customer)\b|"
    r"\b(?:cliente|consumidor|customer)\b.{0,60}\b(?:numero|id|identificador|"
    r"telefone|celular|e-?mail|endereco|cep|documento|number|phone|address|zip|"
    r"card|document)s?\b)"
)
_INTERNAL_INSTRUCTION_TERMS = re.compile(
    r"\b(?:prompt (?:do|de) sistema|system prompt|prompt interno|internal prompt|"
    r"instrucoes (?:internas|ocultas|do sistema)|"
    r"(?:internal|hidden|system) instructions|mensagem (?:do|de) sistema|"
    r"system message|developer message)\b"
)
_EXFILTRATION_ACTIONS = re.compile(
    r"\b(?:qual|quais|what|liste|listar|enumere|enumerar|mostre|mostrar|exiba|exibir|"
    r"copie|copiar|revele|revelar|leia|ler|read|"
    r"acrescente|acrescentar|inclua|incluir|anexe|anexar|insira|inserir|"
    r"imprima|imprimir|responda|responder|answer|envie|enviar|send|"
    r"passe|passa|passar|mande|manda|mandar|envia|apresente|apresentar|"
    r"me (?:da|de)|dar|give|forneca|fornece|fornecer|retorne|retornar|"
    r"exporte|exportar|extraia|extrair|localize|localizar|"
    r"onde|como|achar|ache|encontrar|encontre|descobrir|descubra|procurar|procure|"
    r"buscar|busque|obter|acesse|acessar|consultar|recuperar|diga|dizer|informar|"
    r"which|list|enumerate|show|display|copy|reveal|provide|return|export|extract|"
    r"locate|where|"
    r"how|find|search|lookup|access|get|retrieve|tell)\b"
)
_DIRECT_ACCESS_ACTIONS = re.compile(
    r"\b(?:onde|achar|ache|encontrar|encontre|descobrir|descubra|procurar|procure|"
    r"buscar|busque|obter|acesse|acessar|localize|"
    r"localizar|recuperar|exporte|exportar|extraia|extrair|leia|ler|read|where|find|access|get|"
    r"locate|retrieve|export|extract|search|lookup)\b"
)
_DEFENSIVE_SECURITY_TERMS = re.compile(
    r"\b(?:proteg\w*|protecao|seguranca|segura|seguro|mascarar|rotacionar|"
    r"evitar|prevencao|prevenir|vazamento|armazenar|guardar|autenticacao|"
    r"boas praticas|protect|protection|security|secure|mask|rotate|avoid|prevent|"
    r"leak|store|authentication|best practices)\b"
)
_NEGATED_DISCLOSURE_ACTIONS = re.compile(
    r"\b(?:sem|nao|without|do not|don't)\s+"
    r"(?:revelar|mostrar|expor|listar|copiar|reveal|show|expose|list|copy)\b"
)
_SAFE_SENSITIVITY_CLASSIFICATION = re.compile(
    r"\b(?:a|the) tabela\b[\s\S]{0,100}\b(?:contem|possui|tem|contains|has)\b"
    r"[\s\S]{0,80}\b(?:dados? pessoais?|dados? sensiveis?|personal data|"
    r"sensitive data)\b"
)
_RECORD_EXTRACTION_TERMS = re.compile(
    r"\b(?:mostre|liste|exiba|copie|amostra|registros?|linhas?|valores?|"
    r"show|list|display|copy|sample|records?|rows?|values?)\b"
)
_SENSITIVE_GUESS_PATTERNS = (
    re.compile(
        r"\b(?:adivinhar|adivinhe|chutar|chute|inventar|invente|imaginar|imagine|"
        r"fabricar|fabrique|guess|invent|imagine|fabricate)\b"
    ),
    re.compile(
        r"\b(?:se (?:voce )?tivesse (?:que|de)|qual seria)\b.{0,100}"
        r"\b(?:criar|crie|gerar|gere|create|generate)\b"
    ),
)
_ARBITRARY_DATA_ACCESS_PATTERNS = (
    re.compile(
        r"\b(?:execute|executar|rode|rodar|run)\b[\s\S]{0,120}"
        r"\bselect\b[\s\S]{0,80}(?:\*|\bfrom\b|\btabelas?\b)"
    ),
    re.compile(
        r"\b(?:execute|executar|rode|rodar|run)\b[\s\S]{0,120}"
        r"\b(?:show tables|information schema|information_schema|describe table)\b"
    ),
)
_PROTECTED_OUTPUT_PATTERNS = (
    re.compile(
        r"\b(?:o|este|meu) prompt (?:do|de) sistema(?: completo)? (?:e|eh|is)\b"
    ),
    re.compile(
        r"\b(?:as|estas|minhas) instrucoes (?:ocultas|internas|do sistema) "
        r"(?:sao|incluem)\b"
    ),
    re.compile(r"\b(?:the|my) (?:full )?system prompt is\b"),
)


def _normalize(text: str) -> str:
    """Normalize case and accents without preserving question details."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    normalized = "".join(
        character for character in decomposed if not unicodedata.combining(character)
    )
    # Separators in variable names must not bypass detection, and common
    # leetspeak substitutions must be treated as equivalent to the original text.
    return normalized.replace("_", " ").translate(str.maketrans({"0": "o", "3": "e"}))


def is_disallowed_question(question: str) -> bool:
    """Blocks prompt injection and requests to locate or exfiltrate sensitive data."""
    normalized = _normalize(question)
    if any(pattern.search(normalized) for pattern in _OVERRIDE_PATTERNS):
        return True
    if any(pattern.search(normalized) for pattern in _ARBITRARY_DATA_ACCESS_PATTERNS):
        return True
    contains_sensitive_information = bool(
        _SENSITIVE_TERMS.search(normalized)
        or _OBFUSCATED_SENSITIVE_TERMS.search(normalized)
        or _CUSTOMER_PERSONAL_DATA_TERMS.search(normalized)
        or _INTERNAL_INSTRUCTION_TERMS.search(normalized)
    )
    if contains_sensitive_information and any(
        pattern.search(normalized) for pattern in _SENSITIVE_GUESS_PATTERNS
    ):
        return True
    actionable_question = _NEGATED_DISCLOSURE_ACTIONS.sub("", normalized)
    requests_sensitive_information = bool(
        contains_sensitive_information
        and _EXFILTRATION_ACTIONS.search(actionable_question)
    )
    if not requests_sensitive_information:
        return False
    if _SAFE_SENSITIVITY_CLASSIFICATION.search(
        normalized
    ) and not _RECORD_EXTRACTION_TERMS.search(normalized):
        return False
    if _DEFENSIVE_SECURITY_TERMS.search(
        normalized
    ) and not _DIRECT_ACCESS_ACTIONS.search(normalized):
        return False
    return True


def exposes_protected_instructions(answer: str, system_prompt: str) -> bool:
    """Detects disclosure of the internal prompt as a final containment layer."""
    normalized_answer = _normalize(answer)
    if any(pattern.search(normalized_answer) for pattern in _PROTECTED_OUTPUT_PATTERNS):
        return True

    # One isolated rule may match public documentation. Two long prompt lines
    # in the same answer indicate internal prompt reproduction.
    protected_fragments = {
        _normalize(line.removeprefix("- ").strip())
        for line in system_prompt.splitlines()
        if len(line.removeprefix("- ").strip()) >= 45
    }
    matching_fragments = sum(
        fragment in normalized_answer for fragment in protected_fragments
    )
    return matching_fragments >= 2
