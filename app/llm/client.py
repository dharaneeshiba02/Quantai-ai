"""
app/llm/client.py — Dual-provider LLM client (Groq API + Local Ollama/Qwen) with resilience and fallbacks.

Rules:
  - Only one LLM call runs at a time (enforced via _LLM_SEMAPHORE)
  - temperature=0 for SQL, 0.2 for insight
  - Respects cfg.llm_provider ("auto", "groq", "ollama")
  - Seamlessly falls back to Local Qwen if Groq is unavailable, unauthenticated, or rate-limited
  - Health check verifies reachability of active and fallback providers
  - Never logs API keys, user data, or full prompts at INFO level
"""
from __future__ import annotations

import asyncio
import logging
import time
from functools import lru_cache
from typing import AsyncIterator, Optional, Tuple

import httpx
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings

log = logging.getLogger(__name__)

# Global semaphore: allow 2 concurrent LLM calls for parallel sub-agents
_LLM_SEMAPHORE = asyncio.Semaphore(2)


class LLMHealthError(RuntimeError):
    """Raised when the configured language model is unavailable."""
    pass


def _resolve_provider(model: Optional[str] = None) -> Tuple[str, str, str, str]:
    """
    Resolve (provider_name, model_name, base_url, api_key) based on settings and parameters.
    Returns:
        provider: 'groq' | 'ollama'
        model_name: specific model id
        base_url: OpenAI-compatible API endpoint
        api_key: secret key or dummy token for local server
    """
    cfg = get_settings()
    pref = (cfg.llm_provider or "auto").strip().lower()

    # If explicitly ollama, or auto with no Groq API key provided
    if pref == "ollama" or (pref == "auto" and not cfg.groq_api_key):
        model_name = model or cfg.llm_model
        base_url = f"{cfg.ollama_base_url.rstrip('/')}/v1"
        return "ollama", model_name, base_url, "ollama"

    # Default to Groq
    model_name = model or cfg.groq_model
    base_url = cfg.groq_base_url
    api_key = cfg.groq_api_key
    return "groq", model_name, base_url, api_key


async def health_check() -> dict:
    """
    Verify the configured language model endpoint and model availability.
    Supports both Groq and Local Ollama providers.
    """
    cfg = get_settings()
    provider, active_model, base_url, api_key = _resolve_provider()

    result = {
        "provider": provider,
        "provider_reachable": False,
        "model_available": False,
        "model": active_model,
        "embed_model": cfg.llm_embed_model,
        "error": None,
        "available_models": [],
        "fix": None,
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            if provider == "groq":
                resp = await client.get(
                    f"{cfg.groq_base_url.rstrip('/')}/models",
                    headers={"Authorization": f"Bearer {cfg.groq_api_key}"},
                )
                resp.raise_for_status()
                data = resp.json()
                models = [m["id"] for m in data.get("data", [])]
                result["provider_reachable"] = True
                result["available_models"] = models
                result["model_available"] = active_model in models

                if not result["model_available"]:
                    result["error"] = f"Model '{active_model}' is not available on Groq."
                    result["fix"] = "Check the configured model name or set LLM_PROVIDER=ollama."

            else:  # Ollama local
                resp = await client.get(f"{cfg.ollama_base_url.rstrip('/')}/api/tags")
                resp.raise_for_status()
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                result["provider_reachable"] = True
                result["available_models"] = models
                # Match full or prefix (e.g. qwen2.5-coder:7b or qwen2.5-coder:latest)
                model_base = active_model.split(":")[0]
                result["model_available"] = any(active_model == m or m.startswith(model_base) for m in models)

                if not result["model_available"]:
                    result["error"] = f"Model '{active_model}' not found in local Ollama."
                    result["fix"] = f"Run: ollama pull {active_model}"

    except httpx.HTTPStatusError as exc:
        result["error"] = f"LLM provider returned HTTP {exc.response.status_code}."
        result["fix"] = "Check provider API key and configuration."
    except (httpx.ConnectError, httpx.TimeoutException):
        result["error"] = f"LLM provider ({provider}) is unreachable."
        result["fix"] = (
            "Start Ollama with `ollama serve` or check network connection to Groq."
            if provider == "ollama" else "Check internet access to api.groq.com."
        )
    except Exception as exc:
        result["error"] = f"Health check failed: {type(exc).__name__}"

    return result


def _build_client(
    model: str,
    temperature: float,
    model_kwargs: dict | None = None,
    provider_override: Optional[str] = None,
) -> ChatOpenAI:
    cfg = get_settings()
    provider, resolved_model, base_url, api_key = _resolve_provider(model)

    if provider_override == "ollama":
        resolved_model = model or cfg.llm_fallback_model or cfg.llm_model
        base_url = f"{cfg.ollama_base_url.rstrip('/')}/v1"
        api_key = "ollama"

    return ChatOpenAI(
        model=resolved_model,
        api_key=api_key or "local",
        base_url=base_url,
        temperature=temperature,
        timeout=cfg.llm_timeout_sec,
        model_kwargs=model_kwargs or {},
    )


async def call_llm(
    system_prompt: str,
    user_message: str,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    format: Optional[dict] = None,
) -> str:
    """
    Call the LLM with a system + user message.
    Enforces single-call semaphore. Automatically falls back to local Qwen on error.
    """
    cfg = get_settings()
    temp = temperature if temperature is not None else cfg.llm_temperature_sql
    provider, model_name, _, _ = _resolve_provider(model)

    async with _LLM_SEMAPHORE:
        try:
            return await _do_call(system_prompt, user_message, temp, model_name, format)
        except Exception as exc:
            err_str = str(exc).lower()
            is_oom = any(k in err_str for k in ["out of memory", "oom", "cuda", "killed"])
            is_timeout = isinstance(exc, (TimeoutError, httpx.TimeoutException)) or any(
                marker in err_str for marker in ("timeout", "timed out")
            )
            is_auth_or_rate = any(marker in err_str for marker in ("401", "unauthorized", "429", "rate limit", "quota"))

            # Automatic fallback to local Ollama if Groq failed or local OOM/timeout
            if cfg.llm_auto_fallback:
                if provider == "groq":
                    fallback_model = cfg.llm_fallback_model or cfg.llm_model
                    log.warning(
                        "Groq call failed (%s); falling back to local Ollama model %s",
                        type(exc).__name__,
                        fallback_model,
                    )
                    return await _do_call(
                        system_prompt, user_message, temp, fallback_model, format, provider_override="ollama"
                    )
                elif is_oom or is_timeout:
                    fallback_model = cfg.llm_fallback_model
                    if model_name != fallback_model:
                        log.warning(
                            "Ollama call failed (%s); falling back to smaller local model %s",
                            "OOM" if is_oom else "timeout",
                            fallback_model,
                        )
                        return await _do_call(
                            system_prompt, user_message, temp, fallback_model, format, provider_override="ollama"
                        )
            raise


async def _do_call(
    system_prompt: str,
    user_message: str,
    temperature: float,
    model_name: str,
    format: Optional[dict],
    provider_override: Optional[str] = None,
) -> str:
    client_kwargs: dict = {}
    if format:
        client_kwargs["model_kwargs"] = {
            "response_format": {"type": "json_object"},
        }

    client = _build_client(model_name, temperature, model_kwargs=client_kwargs.get("model_kwargs"), provider_override=provider_override)
    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    t0 = time.perf_counter()
    log.debug("LLM call: model=%s temp=%.1f", model_name, temperature)

    response = await client.ainvoke(messages)
    elapsed = time.perf_counter() - t0
    log.debug("LLM response: %.2fs, %d chars", elapsed, len(response.content))

    content = str(response.content).strip()
    # Strip <think>...</think> chain-of-thought blocks emitted by deepseek-r1
    import re as _re
    content = _re.sub(r"<think>[\s\S]*?</think>", "", content, flags=_re.IGNORECASE).strip()
    return content


async def stream_llm(
    system_prompt: str,
    user_message: str,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
) -> AsyncIterator[str]:
    """
    Stream LLM tokens for insight generation.
    Yields text chunks as they arrive.
    """
    cfg = get_settings()
    temp = temperature if temperature is not None else cfg.llm_temperature_insight
    provider, model_name, _, _ = _resolve_provider(model)

    async with _LLM_SEMAPHORE:
        client = _build_client(model=model_name, temperature=temp)
        messages: list[BaseMessage] = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        async for chunk in client.astream(messages):
            if hasattr(chunk, "content") and chunk.content:
                yield str(chunk.content)


async def embed_text(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using the local embedding model via Ollama.
    Returns a list of embedding vectors.
    """
    cfg = get_settings()
    async with httpx.AsyncClient(timeout=60.0) as client:
        embeddings = []
        for text in texts:
            resp = await client.post(
                f"{cfg.ollama_base_url}/api/embeddings",
                json={"model": cfg.llm_embed_model, "prompt": text},
            )
            resp.raise_for_status()
            embeddings.append(resp.json()["embedding"])
        return embeddings
