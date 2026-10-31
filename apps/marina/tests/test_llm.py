from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from apps.marina.src.llm import call_llm


def _fake_client_with_answer(text):
    fake_client = MagicMock()
    fake_choice = MagicMock()
    fake_choice.message.content = text
    fake_response = MagicMock()
    fake_response.choices = [fake_choice]
    fake_client.serving_endpoints.query.return_value = fake_response
    return fake_client


def test_call_llm_returns_message_content():
    fake_client = _fake_client_with_answer("resposta gerada")

    with patch("apps.marina.src.llm.workspace_client", return_value=fake_client):
        answer = call_llm([{"role": "user", "content": "pergunta"}])

    assert answer == "resposta gerada"


def test_call_llm_raises_when_no_choices_returned():
    fake_client = MagicMock()
    fake_response = MagicMock()
    fake_response.choices = []
    fake_client.serving_endpoints.query.return_value = fake_response

    with patch("apps.marina.src.llm.workspace_client", return_value=fake_client):
        with pytest.raises(RuntimeError):
            call_llm([{"role": "user", "content": "pergunta"}])


def test_call_llm_raises_when_content_is_empty():
    fake_client = _fake_client_with_answer("")

    with patch("apps.marina.src.llm.workspace_client", return_value=fake_client):
        with pytest.raises(RuntimeError):
            call_llm([{"role": "user", "content": "pergunta"}])


def test_call_llm_passes_endpoint_name_through():
    fake_client = _fake_client_with_answer("ok")

    with patch("apps.marina.src.llm.workspace_client", return_value=fake_client):
        call_llm([{"role": "user", "content": "pergunta"}], endpoint="outro-endpoint")

    _, kwargs = fake_client.serving_endpoints.query.call_args
    assert kwargs["name"] == "outro-endpoint"
