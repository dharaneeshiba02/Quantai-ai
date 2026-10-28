"""
tests/test_llm_dual_provider.py — Tests for dual provider LLM client (Groq + Ollama).
"""
from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.llm.client import _resolve_provider, health_check


class LLMDualProviderTests(unittest.TestCase):
    def test_resolve_provider_groq(self):
        mock_settings = MagicMock()
        mock_settings.llm_provider = "auto"
        mock_settings.groq_api_key = "gsk_test123"
        mock_settings.groq_model = "openai/gpt-oss-120b"
        mock_settings.groq_base_url = "https://api.groq.com/openai/v1"

        with patch("app.llm.client.get_settings", return_value=mock_settings):
            provider, model, base_url, api_key = _resolve_provider()
            self.assertEqual(provider, "groq")
            self.assertEqual(model, "openai/gpt-oss-120b")
            self.assertEqual(base_url, "https://api.groq.com/openai/v1")
            self.assertEqual(api_key, "gsk_test123")

    def test_resolve_provider_ollama_explicit_or_missing_key(self):
        mock_settings = MagicMock()
        mock_settings.llm_provider = "ollama"
        mock_settings.groq_api_key = ""
        mock_settings.llm_model = "qwen2.5-coder:7b"
        mock_settings.ollama_base_url = "http://127.0.0.1:11434"

        with patch("app.llm.client.get_settings", return_value=mock_settings):
            provider, model, base_url, api_key = _resolve_provider()
            self.assertEqual(provider, "ollama")
            self.assertEqual(model, "qwen2.5-coder:7b")
            self.assertEqual(base_url, "http://127.0.0.1:11434/v1")
            self.assertEqual(api_key, "ollama")


def test_health_check_groq_healthy():
    import asyncio
    mock_settings = MagicMock()
    mock_settings.llm_provider = "groq"
    mock_settings.groq_api_key = "gsk_valid"
    mock_settings.groq_model = "llama-3.3-70b-versatile"
    mock_settings.groq_base_url = "https://api.groq.com/openai/v1"
    mock_settings.llm_embed_model = "nomic-embed-text"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"data": [{"id": "llama-3.3-70b-versatile"}, {"id": "mixtral-8x7b-32768"}]}

    with patch("app.llm.client.get_settings", return_value=mock_settings), \
         patch("httpx.AsyncClient.get", return_value=mock_resp):
        res = asyncio.run(health_check())
        assert res["provider"] == "groq"
        assert res["provider_reachable"] is True
        assert res["model_available"] is True


def test_health_check_ollama_healthy():
    import asyncio
    mock_settings = MagicMock()
    mock_settings.llm_provider = "ollama"
    mock_settings.groq_api_key = ""
    mock_settings.llm_model = "qwen2.5-coder:7b"
    mock_settings.ollama_base_url = "http://127.0.0.1:11434"
    mock_settings.llm_embed_model = "nomic-embed-text"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"models": [{"name": "qwen2.5-coder:7b"}, {"name": "nomic-embed-text:latest"}]}

    with patch("app.llm.client.get_settings", return_value=mock_settings), \
         patch("httpx.AsyncClient.get", return_value=mock_resp):
        res = asyncio.run(health_check())
        assert res["provider"] == "ollama"
        assert res["provider_reachable"] is True
        assert res["model_available"] is True
