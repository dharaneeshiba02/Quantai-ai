"""
app/accuracy/query_planner.py — Structured query planning via LLM with Pydantic validation.

Uses Ollama structured output (JSON schema) to generate a constrained plan,
with fallback to strict JSON prompting + parse-and-retry.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.llm.client import call_llm
from app.llm.prompts import build_planner_prompt

log = logging.getLogger(__name__)


class MetricSpec(BaseModel):
    column: str
    agg: Literal["sum", "avg", "count", "count_distinct", "min", "max", "median"]


class FilterSpec(BaseModel):
    column: str
    op: str = "="
    value: Any = None


class OrderSpec(BaseModel):
    column: str
    direction: Literal["ASC", "DESC"] = "DESC"


class TimeRangeSpec(BaseModel):
    column: str
    start: Optional[str] = None
    end: Optional[str] = None
    description: Optional[str] = None


class QueryPlan(BaseModel):
    """Validated query plan from the LLM."""
    intent: Literal["aggregate", "top_n", "trend", "distribution", "correlation", "ratio", "follow_up"]
    metrics: list[MetricSpec] = Field(default_factory=list)
    group_by: list[str] = Field(default_factory=list)
    filters: list[FilterSpec] = Field(default_factory=list)
    time_range: Optional[TimeRangeSpec] = None
    order_by: list[OrderSpec] = Field(default_factory=list)
    limit: Optional[int] = 1000
    compare_to: Optional[dict] = None
    confidence: Literal["high", "medium", "low"] = "medium"
    ambiguous_columns: list[str] = Field(default_factory=list)
    ranking_factors: list[dict[str, str]] = Field(default_factory=list)

    @field_validator("metrics")
    @classmethod
    def at_least_one_metric_or_group(cls, v: list, info: Any) -> list:
        # Allow empty metrics for "trend" with group_by
        return v

    def to_dict(self) -> dict:
        return self.model_dump()


PLAN_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": ["aggregate", "top_n", "trend", "distribution", "correlation", "ratio", "follow_up"]
        },
        "metrics": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "agg": {
                        "type": "string",
                        "enum": ["sum", "avg", "count", "count_distinct", "min", "max", "median"]
                    }
                },
                "required": ["column", "agg"]
            }
        },
        "group_by": {"type": "array", "items": {"type": "string"}},
        "filters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "op": {"type": "string"},
                    "value": {}
                },
                "required": ["column"]
            }
        },
        "time_range": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "properties": {
                        "column": {"type": "string"},
                        "start": {"type": "string"},
                        "end": {"type": "string"}
                    }
                }
            ]
        },
        "order_by": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "string"},
                    "direction": {"type": "string", "enum": ["ASC", "DESC"]}
                }
            }
        },
        "limit": {"oneOf": [{"type": "integer"}, {"type": "null"}]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
        "ambiguous_columns": {"type": "array", "items": {"type": "string"}},
        "ranking_factors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "kind": {"type": "string", "enum": ["label", "benefit", "cost", "availability", "popularity"]},
                    "column": {"type": "string"}
                },
                "required": ["kind", "column"]
            }
        }
    },
    "required": ["intent", "metrics", "confidence"]
}


async def generate_plan(
    question: str,
    schema_columns: list[dict[str, Any]],
    resolved_dates: str | None = None,
    chat_history: list[str] | None = None,
    temperature: float = 0.0,
) -> tuple[Optional[QueryPlan], str | None]:
    """
    Generate a structured query plan for the given question.

    Args:
        question: The user's question.
        schema_columns: Selected columns (from schema_selector).
        resolved_dates: Human-readable date context string.
        chat_history: Recent conversation for follow-up context.
        temperature: LLM temperature (0 for deterministic).

    Returns:
        (QueryPlan, None) on success, or (None, error_message) on failure.
    """
    from app.llm.prompts import build_schema_block

    schema_block = build_schema_block(schema_columns)
    system_prompt, user_message = build_planner_prompt(
        question=question,
        schema_block=schema_block,
        resolved_dates=resolved_dates,
        chat_history=chat_history,
    )

    # Try with structured output first
    try:
        raw = await call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=temperature,
            format=PLAN_JSON_SCHEMA,
        )
        plan = _parse_plan(raw)
        if plan:
            return plan, None
    except Exception as exc:
        log.debug("Structured output failed (%s), trying plain JSON", exc)

    # Fallback: plain JSON prompting with parse-and-retry
    for attempt in range(2):
        try:
            raw = await call_llm(
                system_prompt=system_prompt,
                user_message=user_message + "\n\nReturn ONLY the JSON object, no other text:",
                temperature=temperature + (attempt * 0.1),
            )
            plan = _parse_plan(raw)
            if plan:
                return plan, None
        except Exception as exc:
            log.warning("Plan generation attempt %d failed: %s", attempt + 1, exc)

    return None, "Could not generate a structured query plan. Falling back to direct SQL generation."


def _parse_plan(raw: str) -> Optional[QueryPlan]:
    """Parse and validate raw JSON string into a QueryPlan."""
    # Extract JSON from response (may have surrounding text)
    raw = raw.strip()

    # Remove markdown code fences if present
    if "```" in raw:
        import re
        match = re.search(r"```(?:json)?\s*([\s\S]+?)```", raw)
        if match:
            raw = match.group(1).strip()

    # Find the JSON object
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start == -1 or end == 0:
        log.debug("No JSON object found in response: %s", raw[:200])
        return None

    json_str = raw[start:end]
    try:
        data = json.loads(json_str)
        return QueryPlan(**data)
    except (json.JSONDecodeError, Exception) as exc:
        log.debug("Plan parse error: %s — raw: %s", exc, json_str[:200])
        return None
