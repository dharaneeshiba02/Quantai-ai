"""
app/llm/client.py — API chat client with health check and fallback logic.

Rules:
  - Only one LLM call runs at a time (enforced by caller via async)
  - temperature=0 for SQL, 0.2 for insight
  - Falls back to smaller model on timeout or memory error
  - Health check verifies Ollama is reachable and model is available
  - Never logs user data or full prompts at INFO level
"""
from __future__ import annotations

import asyncio
import logging
import time
from functools import lru_cache
from typing import AsyncIterator, Optional

import httpx
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings

log = logging.getLogger(__name__)

# Global semaphore: one LLM call at a time
_LLM_SEMAPHORE = asyncio.Semaphore(1)


class LLMHealthError(RuntimeError):
    """Raised when the configured language model is unavailable."""
    pass


async def health_check() -> dict:
    """
    Verify the configured language model endpoint and model availability.
    Returns a dict with status, available models, and any error message.
    Prints a clear fix message if something is wrong.
    """
    cfg = get_settings()
    result = {
        "provider_reachable": False,
        "model_available": False,
        "model": cfg.groq_model,
        "embed_model": cfg.llm_embed_model,
        "error": None,
        "available_models": [],
        "fix": None,
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(
                f"{cfg.groq_base_url.rstrip('/')}/models",
                headers={"Authorization": f"Bearer {cfg.groq_api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            models = [m["id"] for m in data.get("data", [])]
            result["provider_reachable"] = True
            result["available_models"] = models

            result["model_available"] = cfg.groq_model in models

            if not result["model_available"]:
                result["error"] = f"Model '{cfg.groq_model}' is not available."
                result["fix"] = "Check the configured model name and provider access."

    except httpx.HTTPStatusError as exc:
        result["error"] = f"LLM provider returned HTTP {exc.response.status_code}."
        result["fix"] = "Check the provider configuration and network connection."
    except (httpx.ConnectError, httpx.TimeoutException):
        result["error"] = "LLM provider is unreachable."
        result["fix"] = "Check the provider configuration and network connection."
        log.error("Cannot connect to configured LLM provider")
    except Exception as exc:
        result["error"] = "LLM provider health check failed."
        log.error("Unexpected LLM provider health-check failure: %s", type(exc).__name__)

    return result


def _build_client(
    model: str,
    temperature: float,
    model_kwargs: dict | None = None,
) -> ChatOpenAI:
    cfg = get_settings()
    return ChatOpenAI(
        model=model,
        api_key=cfg.groq_api_key,
        base_url=cfg.groq_base_url,
        temperature=temperature,
        timeout=cfg.llm_timeout_sec,
        model_kwargs=model_kwargs or {},
    )


async def call_llm(
    system_prompt: str,
    user_message: str,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
    format: Optional[dict] = None,  # JSON schema for structured output
) -> str:
    """
    Call the LLM with a system + user message.
    Enforces single-call semaphore. Falls back to smaller model on error.
    
    Args:
        system_prompt: System instructions (never include user data at INFO log level).
        user_message: The user / data portion of the prompt.
        temperature: Override; defaults to cfg.llm_temperature_sql.
        model: Override model name.
        format: Optional JSON schema dict for structured output (Ollama format param).
    
    Returns:
        The model's text response.
    """
    cfg = get_settings()
    temp = temperature if temperature is not None else cfg.llm_temperature_sql
    model_name = model or cfg.groq_model
    fallback_model = cfg.groq_model

    async with _LLM_SEMAPHORE:
        try:
            return await _do_call(system_prompt, user_message, temp, model_name, format)
        except Exception as exc:
            err_str = str(exc).lower()
            is_oom = any(k in err_str for k in ["out of memory", "oom", "cuda", "killed"])
            is_timeout = isinstance(exc, (TimeoutError, httpx.TimeoutException)) or any(
                marker in err_str for marker in ("timeout", "timed out")
            )

            if cfg.llm_auto_fallback and (is_oom or is_timeout) and model_name != fallback_model:
                log.warning(
                    "LLM call failed (%s), falling back to %s",
                    "OOM" if is_oom else "timeout",
                    fallback_model,
                )
                return await _do_call(
                    system_prompt, user_message, temp, fallback_model, format
                )
            raise


async def _do_call(
    system_prompt: str,
    user_message: str,
    temperature: float,
    model_name: str,
    format: Optional[dict],
) -> str:
    cfg = get_settings()
    client_kwargs: dict = {}
    if format:
        client_kwargs["model_kwargs"] = {
            "response_format": {"type": "json_object"},
        }

    client = _build_client(model_name, temperature, **client_kwargs)
    messages: list[BaseMessage] = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message),
    ]

    t0 = time.perf_counter()
    log.debug("LLM call: model=%s temp=%.1f ctx=%d", model_name, temperature, cfg.llm_num_ctx)

    response = await client.ainvoke(messages)
    elapsed = time.perf_counter() - t0
    log.debug("LLM response: %.2fs, %d chars", elapsed, len(response.content))

    return str(response.content).strip()


async def stream_llm(
    system_prompt: str,
    user_message: str,
    temperature: Optional[float] = None,
    model: Optional[str] = None,
) -> AsyncIterator[str]:
    """
    Stream LLM tokens for the insight generation phase.
    Yields text chunks as they arrive.
    """
    cfg = get_settings()
    temp = temperature if temperature is not None else cfg.llm_temperature_insight
    model_name = model or cfg.groq_model

    async with _LLM_SEMAPHORE:
        client = _build_client(
            model=model_name,
            temperature=temp,
        )
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
