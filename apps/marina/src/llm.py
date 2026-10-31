"""Call Marina's LLM through Databricks Model Serving.

The default endpoint recommended by the technical design
(`databricks-meta-llama-3-3-70b-instruct`) can be configured with the
`MARINA_LLM_ENDPOINT` environment variable without changing the code.
"""

from __future__ import annotations

from .config import LLM_ENDPOINT, workspace_client

MAX_TOKENS = 600


def call_llm(
    messages: list[dict[str, str]],
    endpoint: str = LLM_ENDPOINT,
    max_tokens: int = MAX_TOKENS,
) -> str:
    """Calls the configured LLM endpoint and returns the response text."""
    from databricks.sdk.service.serving import ChatMessage, ChatMessageRole

    client = workspace_client()
    chat_messages = [
        ChatMessage(role=ChatMessageRole(message["role"]), content=message["content"])
        for message in messages
    ]
    response = client.serving_endpoints.query(
        name=endpoint,
        messages=chat_messages,
        max_tokens=max_tokens,
    )
    if not response.choices:
        raise RuntimeError(f"Endpoint '{endpoint}' returned no response choices.")
    content = response.choices[0].message.content
    if not content:
        raise RuntimeError(f"Endpoint '{endpoint}' returned an empty response.")
    return content
