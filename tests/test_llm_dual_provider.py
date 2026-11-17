"""Tests for the local-only Ollama LLM client."""
from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from app.llm.client import _resolve_provider, health_check


class LLMDualProviderTests(unittest.TestCase):
    def test_resolve_provider_is_always_local(self):
        mock_settings = MagicMock()
        mock_settings.llm_model = "qwen3.5"
        mock_settings.ollama_base_url = "http://127.0.0.1:11434"

        with patch("app.llm.client.get_settings", return_value=mock_settings):
            provider, model, base_url, api_key = _resolve_provider()
            self.assertEqual(provider, "ollama")
            self.assertEqual(model, "qwen3.5")
            self.assertEqual(base_url, "http://127.0.0.1:11434/v1")
            self.assertEqual(api_key, "ollama")


def test_health_check_ollama_healthy():
    import asyncio
    mock_settings = MagicMock()
    mock_settings.llm_model = "qwen3.5"
    mock_settings.ollama_base_url = "http://127.0.0.1:11434"
    mock_settings.llm_embed_model = "nomic-embed-text"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"models": [{"name": "qwen3.5"}, {"name": "nomic-embed-text:latest"}]}

    with patch("app.llm.client.get_settings", return_value=mock_settings), \
         patch("httpx.AsyncClient.get", return_value=mock_resp):
        res = asyncio.run(health_check())
        assert res["provider"] == "ollama"
        assert res["provider_reachable"] is True
        assert res["model_available"] is True
