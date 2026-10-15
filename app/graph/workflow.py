"""
app/graph/workflow.py — LangGraph workflow definition for QuantAI AI.
Defines the stateful graph with all agent nodes and conditional edges.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from app.graph.state import AgentState

log = logging.getLogger(__name__)

MAX_RETRIES = 2


# ─────────────────────────────────────────────────────────────────────────────
# Routing functions
# ─────────────────────────────────────────────────────────────────────────────

def route_after_supervisor(state: AgentState) -> Literal[
    "source_resolver", "clarification", "end_error"
]:
    intent = state.get("intent", "data_query")
    if state.get("clarification_question") and not state.get("previous_plan"):
        return "clarification"
    if state.get("fatal_error"):
        return "end_error"
    return "source_resolver"


def route_after_execution(state: AgentState) -> Literal[
    "retry_sql", "cross_check", "end_error"
]:
    val = state.get("validation_result") or {}
    retry = state.get("retry_count") or 0

    if not val.get("valid", True) and retry < MAX_RETRIES:
        return "retry_sql"
    elif not val.get("valid", True) and retry >= MAX_RETRIES:
        return "end_error"
    return "cross_check"


def route_after_cross_check(state: AgentState) -> Literal["cleaner", "retry_sql", "cleaner"]:
    cross = state.get("cross_check_result") or {}
    retry = state.get("retry_count") or 0

    # On mismatch and in balanced/max mode and retries remaining, try again
    from app.config import get_settings
    cfg = get_settings()
    if (
        cross
        and cross.get("verified") is False
        and cfg.accuracy_mode in ("balanced", "max")
        and retry < MAX_RETRIES
    ):
        return "retry_sql"

    return "cleaner"  # Proceed regardless (show verification status in UI)


# ─────────────────────────────────────────────────────────────────────────────
# Special nodes
# ─────────────────────────────────────────────────────────────────────────────

async def data_overview_node(state: AgentState) -> dict[str, Any]:
    """Answer dataset structure questions directly from the catalog."""
    from app.graph.state import cache_get
    import re

    catalog = state.get("full_catalog", [])
    source = state.get("selected_source") or {}

    null_cols = [c["safe_name"] for c in catalog if (c.get("null_pct") or 0) > 20]
    date_cols = [c["safe_name"] for c in catalog if c.get("is_date_col")]

    row_count = source.get('row_count', 'unknown')
    row_str = f"{row_count:,}" if isinstance(row_count, int) else str(row_count)

    overview_text = (
        f"## {source.get('name', 'Dataset')}: structure\n\n"
        f"**{row_str} rows · {len(catalog)} columns**\n\n"
        f"- **Date fields:** {', '.join(f'`{name}`' for name in date_cols) or 'none'}\n"
        f"- **Columns with over 20% missing:** {', '.join(f'`{name}`' for name in null_cols) or 'none'}\n\n"
        "### Fields\n"
    )
    wants_all_columns = bool(re.search(r"\b(all|every)\s+(the\s+)?columns\b", state.get("question", "").lower()))
    displayed_catalog = catalog if wants_all_columns else catalog[:24]
    fields = [
        f"`{col['safe_name']}` ({col['col_type']})"
        for col in displayed_catalog
    ]
    overview_text += " · ".join(fields)
    if not wants_all_columns and len(catalog) > len(fields):
        overview_text += f"\n\n_And {len(catalog) - len(fields)} more fields._"

    return {
        "insight_text": overview_text,
        "chart_json": None,
        "generated_sql": None,
        "current_step": "data_overview",
        "trace": state.get("trace", []) + [{
            "step": "data_overview",
            "status": "completed",
            "duration_ms": 0,
        }],
    }


async def clarification_node(state: AgentState) -> dict[str, Any]:
    """Return state with the clarification question set — graph ends here."""
    return {"intent": "clarification_needed"}


async def retry_sql_node(state: AgentState) -> dict[str, Any]:
    """Increment retry counter and set fallback mode for next SQL compile."""
    retry = state.get("retry_count", 0) + 1
    return {
        "retry_count": retry,
        "is_fallback_sql": True,  # force free-form on retry
        "query_plan": None,  # clear plan so sql_compile goes to free-form
        "warnings": state.get("warnings", []) + [f"Retrying SQL generation (attempt {retry + 1})..."],
    }


async def cross_check_node(state: AgentState) -> dict[str, Any]:
    """Run the Pandas cross-check."""
    from app.accuracy.cross_check import cross_check_result
    from app.graph.state import cache_get
    import time

    t0 = time.perf_counter()
    step = "cross_check"
    run_id = state["run_id"]
    plan = state.get("query_plan")

    if not plan:
        return {
            "cross_check_result": {"verified": None, "details": "No plan — skipping cross-check"},
            "verified": False,
            "current_step": step,
        }

    result_df = cache_get(run_id, "result_df")
    full_df = cache_get(run_id, "full_df")  # loaded at source resolution if available

    if result_df is None or full_df is None:
        return {
            "cross_check_result": {"verified": None, "details": "Full data not loaded — skipping cross-check"},
            "verified": False,
            "current_step": step,
        }

    check = cross_check_result(plan, result_df, full_df)
    elapsed = int((time.perf_counter() - t0) * 1000)

    warnings = list(state.get("warnings", []))
    if check.get("verified") is False:
        warnings.append(f"⚠️ Cross-check: SQL vs Pandas mismatch ({check.get('details', '')})")

    return {
        "cross_check_result": check,
        "verified": check.get("verified", False) or False,
        "warnings": warnings,
        "current_step": step,
        "trace": state.get("trace", []) + [{
            "step": step,
            "status": "completed",
            "duration_ms": elapsed,
            "verified": check.get("verified"),
            "details": check.get("details"),
        }],
    }


async def confidence_node(state: AgentState) -> dict[str, Any]:
    """Compute overall confidence label."""
    plan = state.get("query_plan") or {}
    verified = state.get("verified", False)
    cross = state.get("cross_check_result") or {}
    grounding = state.get("grounding_issues", [])
    ambiguous = state.get("ambiguous_columns", [])
    is_fallback = state.get("is_fallback_sql", False)
    errors = state.get("errors", [])

    score = 3  # start at Medium

    if plan and plan.get("confidence") == "high":
        score += 1
    elif plan and plan.get("confidence") == "low":
        score -= 1

    if verified:
        score += 1
    elif cross.get("verified") is False:
        score -= 1

    if grounding:
        score -= len(grounding)

    if ambiguous:
        score -= 1

    if is_fallback:
        score -= 1

    if errors:
        score -= 2

    if score >= 4:
        confidence = "High"
    elif score >= 2:
        confidence = "Medium"
    else:
        confidence = "Low"

    return {"confidence": confidence}


# ─────────────────────────────────────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────────────────────────────────────

def build_workflow() -> Any:
    """Build and compile the QuantAI AI LangGraph workflow."""
    from app.agents.supervisor import supervisor_node
    from app.agents.source_resolver import source_resolver_node
    from app.agents.sql_generator import (
        column_selector_node,
        query_planner_node,
        sql_compile_node,
        validate_and_execute_node,
    )
    from app.agents.cleaner import cleaner_node
    from app.agents.stats_analyzer import stats_analyzer_node
    from app.agents.visualizer import visualizer_node
    from app.agents.insight import insight_node
    from app.agents.data_summary import data_summary_node

    graph = StateGraph(AgentState)

    # ── Add nodes ────────────────────────────────────────────────────────────
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("source_resolver", source_resolver_node)
    graph.add_node("data_overview", data_overview_node)
    graph.add_node("data_summary", data_summary_node)
    graph.add_node("clarification", clarification_node)
    graph.add_node("column_selector", column_selector_node)
    graph.add_node("query_planner", query_planner_node)
    graph.add_node("sql_compile", sql_compile_node)
    graph.add_node("validate_execute", validate_and_execute_node)
    graph.add_node("retry_sql", retry_sql_node)
    graph.add_node("cross_check", cross_check_node)
    graph.add_node("cleaner", cleaner_node)
    graph.add_node("stats_analyzer", stats_analyzer_node)
    graph.add_node("visualizer", visualizer_node)
    graph.add_node("insight", insight_node)
    graph.add_node("confidence", confidence_node)

    # ── Edges ────────────────────────────────────────────────────────────────
    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges("supervisor", route_after_supervisor, {
        "source_resolver": "source_resolver",
        "clarification": "clarification",
        "end_error": END,
    })

    def route_after_source_resolver(state: AgentState):
        if state.get("intent") == "data_summary":
            return "data_summary"
        return "column_selector"

    graph.add_conditional_edges("source_resolver", route_after_source_resolver, {
        "data_summary": "data_summary",
        "column_selector": "column_selector",
    })

    # After column_selector: check if we need data_overview or SQL path
    def route_after_source(state: AgentState):
        if state.get("intent") == "data_overview":
            return "data_overview"
        if state.get("clarification_question"):
            return "clarification"
        return "query_planner"

    graph.add_conditional_edges("column_selector", route_after_source, {
        "data_overview": "data_overview",
        "clarification": "clarification",
        "query_planner": "query_planner",
    })

    graph.add_edge("query_planner", "sql_compile")
    graph.add_edge("sql_compile", "validate_execute")

    graph.add_conditional_edges("validate_execute", route_after_execution, {
        "retry_sql": "retry_sql",
        "cross_check": "cross_check",
        "end_error": "insight",  # still try to generate a helpful error insight
    })

    graph.add_edge("retry_sql", "sql_compile")  # loop back for retry

    graph.add_conditional_edges("cross_check", route_after_cross_check, {
        "retry_sql": "retry_sql",
        "cleaner": "cleaner",
    })

    graph.add_edge("cleaner", "stats_analyzer")
    graph.add_edge("stats_analyzer", "visualizer")
    graph.add_edge("visualizer", "insight")
    graph.add_edge("insight", "confidence")
    graph.add_edge("confidence", END)

    # Terminal nodes
    graph.add_edge("data_overview", END)
    graph.add_edge("data_summary", END)
    graph.add_edge("clarification", END)

    return graph.compile()


# Singleton compiled graph
_workflow = None


def get_workflow():
    global _workflow
    if _workflow is None:
        _workflow = build_workflow()
    return _workflow
