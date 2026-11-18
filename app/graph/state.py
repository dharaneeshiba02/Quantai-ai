"""
app/graph/state.py — LangGraph typed state for the QuantAI AI workflow.

DataFrames are NOT stored in state; they live in a separate in-memory cache
keyed by run_id so the LLM-facing state stays small.
"""
from __future__ import annotations

import uuid
from typing import Any, Annotated, Optional
from typing_extensions import TypedDict


# ─────────────────────────────────────────────────────────────────────────────
# DataFrame / result cache (never serialized, never sent to LLM)
# ─────────────────────────────────────────────────────────────────────────────
_RESULT_CACHE: dict[str, Any] = {}


def cache_put(run_id: str, key: str, value: Any) -> None:
    _RESULT_CACHE.setdefault(run_id, {})[key] = value


def cache_get(run_id: str, key: str, default: Any = None) -> Any:
    return _RESULT_CACHE.get(run_id, {}).get(key, default)


def cache_clear(run_id: str) -> None:
    _RESULT_CACHE.pop(run_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# LangGraph state schema
# ─────────────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # ── Input ───────────────────────────────────────────────────────────────
    run_id: str
    question: str
    session_id: Optional[str]
    source_id: Optional[str]
    selected_model: Optional[str]
    chat_history: list[str]          # recent Q&A strings for context

    # ── Routing ─────────────────────────────────────────────────────────────
    intent: Optional[str]            # data_query|data_overview|follow_up|...
    requires_sql: bool
    clarification_question: Optional[str]

    # ── Source / Schema ──────────────────────────────────────────────────────
    selected_source: Optional[dict[str, Any]]   # source metadata
    table_name: Optional[str]
    schema_name: Optional[str]
    full_catalog: list[dict[str, Any]]          # all column catalog entries
    selected_columns: list[dict[str, Any]]      # top-K selected for LLM
    column_catalog_map: dict[str, dict[str, Any]]  # safe_name -> catalog entry

    # ── Query Plan ───────────────────────────────────────────────────────────
    query_plan: Optional[dict[str, Any]]        # validated QueryPlan dict
    interpretation_text: Optional[str]          # plain-English plan description
    grounding_issues: list[dict[str, Any]]
    resolved_date_description: Optional[str]
    ambiguous_columns: list[str]

    # ── SQL ──────────────────────────────────────────────────────────────────
    generated_sql: Optional[str]
    validation_result: Optional[dict[str, Any]]
    is_fallback_sql: bool
    retry_count: int
    sql_error: Optional[str]

    # ── Results ──────────────────────────────────────────────────────────────
    # DataFrames stored in cache, referenced by run_id keys
    result_row_count: Optional[int]
    result_columns: list[str]

    # ── Cross-check ──────────────────────────────────────────────────────────
    cross_check_result: Optional[dict[str, Any]]
    verified: bool

    # ── Analysis ─────────────────────────────────────────────────────────────
    stats: Optional[dict[str, Any]]
    computed_facts: Optional[dict[str, Any]]

    # ── Visualization ────────────────────────────────────────────────────────
    chart_json: Optional[dict[str, Any]]
    chart_type: Optional[str]

    # ── Insight ──────────────────────────────────────────────────────────────
    insight_text: Optional[str]
    number_check_result: Optional[dict[str, Any]]

    # ── Confidence / Quality ─────────────────────────────────────────────────
    confidence: Optional[str]       # High | Medium | Low
    warnings: list[str]

    # ── Trace / Errors ───────────────────────────────────────────────────────
    trace: list[dict[str, Any]]     # step logs with timing
    errors: list[str]
    current_step: Optional[str]

    # ── Follow-up ────────────────────────────────────────────────────────────
    previous_plan: Optional[dict[str, Any]]


def new_state(
    question: str,
    session_id: Optional[str] = None,
    source_id: Optional[str] = None,
    selected_model: Optional[str] = None,
    chat_history: Optional[list[str]] = None,
    previous_plan: Optional[dict[str, Any]] = None,
) -> AgentState:
    """Create a fresh AgentState with defaults."""
    return AgentState(
        run_id=str(uuid.uuid4()),
        question=question,
        session_id=session_id,
        source_id=source_id,
        selected_model=selected_model,
        chat_history=chat_history or [],
        intent=None,
        requires_sql=True,
        clarification_question=None,
        selected_source=None,
        table_name=None,
        schema_name=None,
        full_catalog=[],
        selected_columns=[],
        column_catalog_map={},
        query_plan=None,
        interpretation_text=None,
        grounding_issues=[],
        resolved_date_description=None,
        ambiguous_columns=[],
        generated_sql=None,
        validation_result=None,
        is_fallback_sql=False,
        retry_count=0,
        sql_error=None,
        result_row_count=None,
        result_columns=[],
        cross_check_result=None,
        verified=False,
        stats=None,
        computed_facts=None,
        chart_json=None,
        chart_type=None,
        insight_text=None,
        number_check_result=None,
        confidence=None,
        warnings=[],
        trace=[],
        errors=[],
        current_step=None,
        previous_plan=previous_plan,
    )
