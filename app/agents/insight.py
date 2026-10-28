"""
app/agents/insight.py — Insight generation agent.
Sends only computed facts to LLM, streams text, then runs number checker.
Falls back to deterministic template if numbers don't check out.
"""
from __future__ import annotations

import logging
import time
from typing import Any

from app.graph.state import AgentState, cache_get
from app.accuracy.number_checker import (
    check_insight_numbers,
    generate_template_insight,
)

log = logging.getLogger(__name__)


async def insight_node(state: AgentState) -> dict[str, Any]:
    """Generate a grounded plain-English insight."""
    t0 = time.perf_counter()
    step = "insight"
    run_id = state["run_id"]
    question = state.get("question", "")
    computed_facts = state.get("computed_facts") or cache_get(run_id, "computed_facts") or {}

    try:
        df = cache_get(run_id, "result_df")
        if df is None or df.empty:
            elapsed = int((time.perf_counter() - t0) * 1000)
            return {
                "insight_text": "No data returned for this query.",
                "current_step": step,
                "trace": state.get("trace", []) + [{"step": step, "status": "skipped_empty", "duration_ms": elapsed}],
            }

        # Build result preview (more rows for richer analytical context)
        preview_rows = df.head(10).to_dict("records")
        preview_str = _format_preview(preview_rows)

        from app.llm.client import call_llm
        from app.llm.prompts import build_insight_prompt

        system_prompt, user_message = build_insight_prompt(
            question=question,
            computed_facts=computed_facts,
            result_preview=preview_str,
            result_shape=(len(df), len(df.columns)),
        )

        # Generate insight
        insight_text = await call_llm(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.2,  # slight creativity for natural language
        )
        insight_text = str(insight_text or "").strip()

        # Number check
        check_result = check_insight_numbers(insight_text, computed_facts)

        if not check_result["valid"] and check_result["invented_numbers"]:
            log.warning(
                "Number checker blocked insight (invented: %s). Regenerating...",
                check_result["invented_numbers"],
            )
            # One regeneration attempt at lower temperature
            insight_text = await call_llm(
                system_prompt=system_prompt,
                user_message=user_message + "\n\nIMPORTANT: Use ONLY the exact numbers from the computed facts provided.",
                temperature=0.0,
            )
            insight_text = str(insight_text or "").strip()
            check_result = check_insight_numbers(insight_text, computed_facts)

            # Final check — fall back to template
            if not check_result["valid"]:
                log.warning("Still failing number check. Using template fallback.")
                insight_text = generate_template_insight(question, computed_facts, preview_rows)
                check_result["used_template"] = True

        # A blank model response is still a failed insight. Keep the answer
        # area useful even when the chart and result table were generated.
        if not insight_text:
            insight_text = generate_template_insight(question, computed_facts, preview_rows)
            check_result = {"valid": True, "invented_numbers": [], "used_template": True}

        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "insight_text": insight_text,
            "number_check_result": check_result,
            "current_step": step,
            "trace": state.get("trace", []) + [{
                "step": step,
                "status": "completed",
                "duration_ms": elapsed,
                "number_check_valid": check_result["valid"],
            }],
        }

    except Exception as exc:
        log.exception("Insight node error")
        elapsed = int((time.perf_counter() - t0) * 1000)
        # Fallback to template
        computed_facts = state.get("computed_facts") or {}
        df = cache_get(run_id, "result_df")
        preview_rows = df.head(3).to_dict("records") if df is not None and not df.empty else []
        fallback = generate_template_insight(question, computed_facts, preview_rows)
        return {
            "insight_text": fallback,
            "current_step": step,
            "warnings": state.get("warnings", []) + [f"Insight generation warning: {exc}"],
            "trace": state.get("trace", []) + [{"step": step, "status": "warning", "duration_ms": elapsed}],
        }


def _format_preview(rows: list[dict]) -> str:
    if not rows:
        return "(empty)"
    lines = []
    for i, row in enumerate(rows[:5]):
        line = ", ".join(f"{k}: {v}" for k, v in list(row.items())[:5])
        lines.append(f"Row {i+1}: {line}")
    return "\n".join(lines)
