"""
app/agents/sql_generator.py — SQL generation agent.
Primary path: structured plan → compiled SQL.
Fallback: free-form SQL generation with retry loop.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

import pandas as pd

from app.graph.state import AgentState, cache_put

log = logging.getLogger(__name__)


async def column_selector_node(state: AgentState) -> dict[str, Any]:
    """Select top-K relevant columns for the question using embeddings."""
    t0 = time.perf_counter()
    step = "column_selector"

    try:
        from app.llm.schema_selector import select_columns
        from app.config import get_settings

        cfg = get_settings()
        full_catalog = state.get("full_catalog", [])
        question = state["question"]
        source_id = state.get("source_id") or "unknown"

        selected = await select_columns(
            question=question,
            columns=full_catalog,
            source_id=source_id,
            top_k=cfg.column_selector_top_k,
        )

        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "selected_columns": selected,
            "current_step": step,
            "trace": state.get("trace", []) + [{
                "step": step,
                "status": "completed",
                "duration_ms": elapsed,
                "selected": len(selected),
                "total": len(full_catalog),
            }],
        }

    except Exception as exc:
        log.warning("Column selector failed: %s — using all columns", exc)
        elapsed = int((time.perf_counter() - t0) * 1000)
        # Fall back to full catalog (may exceed prompt budget on wide tables)
        return {
            "selected_columns": state.get("full_catalog", [])[:30],
            "current_step": step,
            "warnings": state.get("warnings", []) + [f"Column selection failed: {exc}"],
            "trace": state.get("trace", []) + [{"step": step, "status": "warning", "duration_ms": elapsed}],
        }


async def query_planner_node(state: AgentState) -> dict[str, Any]:
    """Run the query planner to get a structured JSON plan."""
    t0 = time.perf_counter()
    step = "query_planner"

    question = state["question"]
    selected_columns = state.get("selected_columns", [])
    chat_history = state.get("chat_history", [])
    previous_plan = state.get("previous_plan")

    # Handle follow-up: modify previous plan
    if state.get("intent") == "follow_up" and previous_plan:
        modified = await _modify_plan_for_followup(question, previous_plan, selected_columns)
        if modified:
            elapsed = int((time.perf_counter() - t0) * 1000)
            return {
                "query_plan": modified,
                "current_step": step,
                "trace": state.get("trace", []) + [{
                    "step": step,
                    "status": "completed",
                    "duration_ms": elapsed,
                    "note": "Follow-up: modified previous plan",
                }],
            }

    # Resolve dates
    resolved_dates = None
    from app.accuracy.grounding import resolve_date_phrase
    import re as _re
    date_phrases = [
        "last quarter", "this year", "last year", "this quarter",
        "last 30 days", "last 6 months", "last 12 months",
    ]
    q_lower = question.lower()
    for phrase in date_phrases:
        if phrase in q_lower:
            start, end, desc = resolve_date_phrase(phrase)
            if start:
                resolved_dates = desc
                break

    try:
        from app.accuracy.query_planner import generate_plan
        plan, error = await generate_plan(
            question=question,
            schema_columns=selected_columns,
            resolved_dates=resolved_dates,
            chat_history=chat_history,
        )

        elapsed = int((time.perf_counter() - t0) * 1000)

        if plan:
            if plan.intent == "top_n" and not re.search(r"\b(?:top|best|worst|lowest|highest)\s+\d+\b", question.lower()):
                plan.limit = 1
            if re.search(r"\b(all|every)\s+(the\s+)?columns\b", question.lower()):
                plan.include_all_columns = True
            # Generate interpretation text
            from app.accuracy.plan_to_sql import plan_to_interpretation
            interp = plan_to_interpretation(
                plan.to_dict(),
                state.get("table_name") or "data",
            )

            # Check for ambiguous columns
            ambiguous = plan.ambiguous_columns or []
            clarification = None
            if ambiguous:
                clarification = (
                    f"I found multiple similar columns: {', '.join(ambiguous)}. "
                    "Which one did you mean? Or should I use all of them?"
                )

            return {
                "query_plan": plan.to_dict(),
                "interpretation_text": interp,
                "resolved_date_description": resolved_dates,
                "ambiguous_columns": ambiguous,
                "clarification_question": clarification if ambiguous else state.get("clarification_question"),
                "current_step": step,
                "trace": state.get("trace", []) + [{
                    "step": step,
                    "status": "completed",
                    "duration_ms": elapsed,
                    "intent": plan.intent,
                    "confidence": plan.confidence,
                    "ambiguous": ambiguous,
                }],
            }
        else:
            return {
                "query_plan": None,
                "current_step": step,
                "warnings": state.get("warnings", []) + [error or "Plan generation failed"],
                "trace": state.get("trace", []) + [{"step": step, "status": "fallback", "duration_ms": elapsed}],
            }

    except Exception as exc:
        log.exception("Query planner node error")
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "query_plan": None,
            "current_step": step,
            "errors": state.get("errors", []) + [str(exc)],
            "trace": state.get("trace", []) + [{"step": step, "status": "error", "duration_ms": elapsed}],
        }


async def _modify_plan_for_followup(
    question: str,
    previous_plan: dict[str, Any],
    selected_columns: list[dict],
) -> dict | None:
    """Modify a previous plan for a follow-up question without calling LLM when possible."""
    q_lower = question.lower()
    plan = dict(previous_plan)

    # "top N" modification
    import re
    m = re.search(r"\btop\s+(\d+)\b", q_lower)
    if m:
        plan["limit"] = int(m.group(1))
        return plan

    # "for YEAR" modification
    m = re.search(r"\b(20\d\d)\b", q_lower)
    if m:
        yr = int(m.group(1))
        if plan.get("time_range"):
            plan["time_range"]["start"] = f"{yr}-01-01"
            plan["time_range"]["end"] = f"{yr}-12-31"
            plan["time_range"]["description"] = f"Year {yr}"
            return plan

    # "by month" modification
    if "by month" in q_lower and plan.get("time_range"):
        date_col = plan["time_range"]["column"]
        plan["group_by"] = [f"DATE_FORMAT(`{date_col}`, '%Y-%m')"]
        plan["limit"] = None
        return plan

    return None


async def sql_compile_node(state: AgentState) -> dict[str, Any]:
    """Compile plan to SQL (primary) or generate free-form SQL (fallback)."""
    t0 = time.perf_counter()
    step = "sql_compile"

    plan = state.get("query_plan")
    table_name = state.get("table_name") or ""
    schema_name = state.get("schema_name", "quantai_data")
    selected_columns = state.get("selected_columns", [])
    question = state["question"]
    sql_error = state.get("sql_error")  # previous error for retry
    retry_count = state.get("retry_count", 0)

    try:
        if plan and not state.get("is_fallback_sql"):
            # Primary: deterministic compiler
            from app.accuracy.plan_to_sql import compile_plan
            from app.accuracy.grounding import ground_plan_filters, resolve_time_range_from_plan

            # Ground filter values
            catalog_map = state.get("column_catalog_map", {})
            grounded_plan, grounding_issues = ground_plan_filters(plan, catalog_map)

            # Resolve date phrases in plan
            import datetime
            data_max_date = None
            max_str = catalog_map.get(
                next((c["safe_name"] for c in state.get("full_catalog", []) if c.get("is_date_col")), ""),
                {}
            ).get("col_max")
            if max_str:
                try:
                    data_max_date = datetime.date.fromisoformat(str(max_str)[:10])
                except Exception:
                    pass

            resolved_tr = resolve_time_range_from_plan(
                grounded_plan.get("time_range"), data_max_date
            )
            if resolved_tr:
                grounded_plan["time_range"] = resolved_tr

            sql = compile_plan(
                grounded_plan,
                table_name,
                schema_name,
                all_columns=bool(grounded_plan.get("include_all_columns")),
            )
            is_fallback = False

            elapsed = int((time.perf_counter() - t0) * 1000)
            return {
                "generated_sql": sql,
                "is_fallback_sql": False,
                "grounding_issues": grounding_issues,
                "current_step": step,
                "trace": state.get("trace", []) + [{
                    "step": step,
                    "status": "completed",
                    "duration_ms": elapsed,
                    "method": "compiled",
                }],
            }

        else:
            # Fallback: free-form SQL generation
            from app.llm.client import call_llm
            from app.llm.prompts import build_sql_prompt, build_schema_block

            schema_block = build_schema_block(selected_columns)
            system_prompt, user_message = build_sql_prompt(
                question=question,
                table_name=table_name,
                schema_block=schema_block,
                error_context=sql_error,
            )

            raw_sql = await call_llm(
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=0.0,
            )

            # Clean markdown fences
            raw_sql = _clean_sql(raw_sql)

            elapsed = int((time.perf_counter() - t0) * 1000)
            return {
                "generated_sql": raw_sql,
                "is_fallback_sql": True,
                "warnings": state.get("warnings", []) + ["Using free-form SQL generation (lower confidence)."],
                "current_step": step,
                "trace": state.get("trace", []) + [{
                    "step": step,
                    "status": "completed",
                    "duration_ms": elapsed,
                    "method": "free-form",
                    "attempt": retry_count + 1,
                }],
            }

    except Exception as exc:
        log.exception("SQL compile node error")
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "errors": state.get("errors", []) + [f"SQL compile failed: {exc}"],
            "current_step": step,
            "trace": state.get("trace", []) + [{"step": step, "status": "error", "duration_ms": elapsed}],
        }


def _clean_sql(raw: str) -> str:
    """Strip markdown code fences and extract SQL."""
    raw = raw.strip()
    if "```" in raw:
        m = re.search(r"```(?:sql)?\s*([\s\S]+?)```", raw)
        if m:
            return m.group(1).strip()
    return raw


async def validate_and_execute_node(state: AgentState) -> dict[str, Any]:
    """Validate SQL through sql_guard then execute it."""
    t0 = time.perf_counter()
    step = "validate_execute"

    sql = state.get("generated_sql") or ""
    table_name = state.get("table_name") or ""
    full_catalog = state.get("full_catalog", [])
    run_id = state["run_id"]

    # Build known_tables for validation
    known_tables: dict[str, list[str]] = {table_name: [c["safe_name"] for c in full_catalog]} if table_name else {}

    try:
        from app.security.sql_guard import validate_sql
        guard = validate_sql(sql, known_tables=known_tables)

        if not guard.valid:
            elapsed = int((time.perf_counter() - t0) * 1000)
            return {
                "validation_result": {"valid": False, "error": guard.error},
                "sql_error": guard.error,
                "current_step": step,
                "trace": state.get("trace", []) + [{
                    "step": step,
                    "status": "validation_failed",
                    "duration_ms": elapsed,
                    "error": guard.error,
                }],
            }

        # Execute with read-only user
        import pandas as pd
        from app.storage.database import readonly_session
        from sqlalchemy import text

        with readonly_session() as conn:
            result_df = pd.read_sql(text(guard.sql), conn)

        # Cache the result DataFrame
        cache_put(run_id, "result_df", result_df)

        elapsed = int((time.perf_counter() - t0) * 1000)
        warnings = list(state.get("warnings", []))
        warnings.extend([f"⚠️ {w}" for w in guard.warnings])

        # Sanity check: warn on unexpected results
        sanity_warnings = _sanity_check(result_df, state)
        warnings.extend(sanity_warnings)

        return {
            "generated_sql": guard.sql,  # use normalised SQL
            "validation_result": {"valid": True, "warnings": guard.warnings},
            "result_row_count": len(result_df),
            "result_columns": list(result_df.columns),
            "sql_error": None,
            "warnings": warnings,
            "current_step": step,
            "trace": state.get("trace", []) + [{
                "step": step,
                "status": "completed",
                "duration_ms": elapsed,
                "rows": len(result_df),
                "cols": len(result_df.columns),
            }],
        }

    except Exception as exc:
        log.warning("Execution error: %s", exc)
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "validation_result": {"valid": False, "error": str(exc)},
            "sql_error": str(exc),
            "current_step": step,
            "trace": state.get("trace", []) + [{
                "step": step,
                "status": "error",
                "duration_ms": elapsed,
                "error": str(exc),
            }],
        }


def _sanity_check(df: pd.DataFrame, state: AgentState) -> list[str]:
    """Run basic sanity checks on the result DataFrame."""
    warnings = []
    if df.empty:
        plan = state.get("query_plan") or {}
        filters = plan.get("filters", [])
        grounding = state.get("grounding_issues", [])
        if grounding:
            warnings.append(
                f"Empty result — a filter value may not match the data: "
                f"{grounding[0].get('message', '')}"
            )
        elif filters:
            warnings.append("Empty result — the filter conditions may not match any rows in the data.")
        else:
            warnings.append("Empty result — the data may not have records for the specified criteria.")

    return warnings
