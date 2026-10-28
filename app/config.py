"""
app/config.py — Central configuration via pydantic-settings.
All settings come from environment variables / .env file.
Credentials NEVER reach the LLM or appear in logs.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Silence LangChain/LangSmith tracing unconditionally before any import
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("LANGCHAIN_API_KEY", "disabled")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "false")


class Settings(BaseSettings):
    """All application settings — loaded from .env or environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ─────────────────────────────────────────────────────────────────
    ollama_base_url: str = "http://127.0.0.1:11434"
    llm_model: str = "qwen2.5-coder:7b"
    llm_fallback_model: str = "openai/gpt-oss-20b"      # Groq fallback if primary unavailable
    llm_provider: str = "auto"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"              # Primary Groq reasoning model
    llm_embed_model: str = "nomic-embed-text"
    llm_num_ctx: int = 8192
    llm_temperature_sql: float = 0.0
    llm_temperature_insight: float = 0.15   # Lower = more factual, fewer hallucinations
    llm_timeout_sec: int = 180
    llm_auto_fallback: bool = True

    # ── MySQL ────────────────────────────────────────────────────────────────
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3306
    mysql_root_password: str = "YourPassword"

    mysql_readonly_user: str = "quantai_readonly"
    mysql_readonly_password: str = "YourPassword"

    mysql_import_user: str = "quantai_import"
    mysql_import_password: str = "YourPassword"

    mysql_app_user: str = "quantai_app"
    mysql_app_password: str = "YourPassword"

    mysql_data_schema: str = "quantai_data"
    mysql_app_schema: str = "quantai_app"

    # ── Privacy / Accuracy ───────────────────────────────────────────────────
    schema_only_mode: bool = False
    pii_masking_enabled: bool = True
    pii_column_patterns: str = "name,email,phone,ssn,address,dob,birth"
    sample_rows_for_llm: int = 3
    date_anchor: Literal["data_max", "today"] = "data_max"

    # ── Column Selector ──────────────────────────────────────────────────────
    column_selector_top_k: int = 20
    embedding_cache_dir: Path = Path("data/embeddings")

    # ── Query / Execution ────────────────────────────────────────────────────
    max_query_rows_display: int = 1000
    max_export_rows: int = 100_000
    query_timeout_ms: int = 30_000
    accuracy_mode: Literal["fast", "balanced", "max"] = "balanced"

    # ── Exports / Uploads ────────────────────────────────────────────────────
    exports_dir: Path = Path("data/exports")
    exports_auto_expiry_days: int = 7
    uploads_dir: Path = Path("data/uploads")
    dictionaries_dir: Path = Path("data/dictionaries")

    # ── App ──────────────────────────────────────────────────────────────────
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    frontend_dist: Path = Path("frontend/dist")
    secret_key: str = "change_me_to_random_32_char_string"

    # ── Telemetry (always off) ───────────────────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_api_key: str = "disabled"
    anonymized_telemetry: bool = False

    # ── Derived helpers ──────────────────────────────────────────────────────

    @property
    def pii_patterns_list(self) -> list[str]:
        return [p.strip().lower() for p in self.pii_column_patterns.split(",") if p.strip()]

    @property
    def readonly_dsn(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_readonly_user}:{self.mysql_readonly_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_data_schema}"
            f"?charset=utf8mb4"
        )

    @property
    def import_dsn(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_import_user}:{self.mysql_import_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_data_schema}"
            f"?charset=utf8mb4"
        )

    @property
    def app_dsn(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_app_user}:{self.mysql_app_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_app_schema}"
            f"?charset=utf8mb4"
        )

    @property
    def root_dsn(self) -> str:
        return (
            f"mysql+pymysql://root:{self.mysql_root_password}"
            f"@{self.mysql_host}:{self.mysql_port}"
            f"?charset=utf8mb4"
        )

    @model_validator(mode="after")
    def _ensure_dirs(self) -> "Settings":
        for d in [
            self.embedding_cache_dir,
            self.exports_dir,
            self.uploads_dir,
            self.dictionaries_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
