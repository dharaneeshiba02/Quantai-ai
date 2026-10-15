"""
app/agents/supervisor.py — Supervisor agent: routes questions and handles ambiguity.
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from app.graph.state import AgentState
from app.llm.client import call_llm
from app.llm.prompts import SUPERVISOR_SYSTEM

log = logging.getLogger(__name__)


async def supervisor_node(state: AgentState) -> dict[str, Any]:
    """
    Route the user question to the appropriate intent.
    Returns state updates.
    """
    t0 = time.perf_counter()
    step = "supervisor"

    question = state["question"]
    chat_history = state.get("chat_history", [])

    # A broad "what does this dataset tell us?" request needs data analysis,
    # not a schema dump. Route it to a compact factual profile.
    q_lower = re.sub(r"\s+", " ", question.lower()).strip()
    summary_phrases = (
        "what does this dataset tell",
        "what does the dataset tell",
        "what does this data tell",
        "what can you tell me about this dataset",
        "tell me about this dataset",
        "summarize this dataset",
        "summarize the dataset",
        "summary of this dataset",
        "dataset summary",
        "dataset overview",
        "overview of this dataset",
        "what is in this dataset",
        "tell me what is in this dataset",
        "give me a summary",
        "give me insights",
        "overall insights",
        "analyze this dataset",
        "analyse this dataset",
        "analyze the data",
        "analyse the data",
        "high-level overview",
    )
    if any(phrase in q_lower for phrase in summary_phrases):
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "intent": "data_summary",
            "requires_sql": False,
            "clarification_question": None,
            "current_step": step,
            "trace": state.get("trace", []) + [{
                "step": step,
                "status": "completed",
                "duration_ms": elapsed,
                "note": "Detected broad dataset summary request",
            }],
        }

    # Only explicit structure/quality requests should return a schema overview.
    overview_keywords = [
        "what columns", "list columns", "show columns", "column names", "what tables", "list tables",
        "column types", "how many rows", "row count", "show schema",
        "schema", "missing values", "null values", "data quality",
        "dataset structure", "overview of columns",
    ]
    if any(kw in q_lower for kw in overview_keywords):
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "intent": "data_overview",
            "requires_sql": False,
            "clarification_question": None,
            "current_step": step,
            "trace": state.get("trace", []) + [{
                "step": step,
                "status": "completed",
                "duration_ms": elapsed,
                "note": "Detected data overview question (no LLM needed)",
            }],
        }

    if re.search(r"\b(?:top|best|worst|lowest|highest)(?:\s+\d+)?\b", q_lower):
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "intent": "data_query",
            "requires_sql": True,
            "clarification_question": None,
            "current_step": step,
            "trace": state.get("trace", []) + [{
                "step": step,
                "status": "completed",
                "duration_ms": elapsed,
                "note": "Detected explicit ranked-list request",
            }],
        }

    # Check for follow-up patterns
    followup_keywords = ["and for", "only the top", "break that down", "filter by",
                         "now show", "same but", "also show", "what about"]
    if state.get("previous_plan") and any(kw in q_lower for kw in followup_keywords):
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "intent": "follow_up",
            "requires_sql": True,
            "clarification_question": None,
            "current_step": step,
            "trace": state.get("trace", []) + [{
                "step": step,
                "status": "completed",
                "duration_ms": elapsed,
                "note": "Detected follow-up question",
            }],
        }

    # Call LLM for routing
    history_ctx = ""
    if chat_history:
        history_ctx = f"\nRecent conversation:\n" + "\n".join(chat_history[-3:])

    user_msg = f"Question: {question}{history_ctx}"
    if state.get("source_id"):
        user_msg += f"\nActive source_id: {state['source_id']}"

    try:
        raw = await call_llm(
            system_prompt=SUPERVISOR_SYSTEM,
            user_message=user_msg,
            temperature=0.0,
        )

        # Parse JSON response
        raw = raw.strip()
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start != -1 and end > start:
            data = json.loads(raw[start:end])
        else:
            data = {"intent": "data_query", "requires_sql": True, "clarification_question": None}

        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "intent": data.get("intent", "data_query"),
            "requires_sql": data.get("requires_sql", True),
            "clarification_question": data.get("clarification_question"),
            "current_step": step,
            "trace": state.get("trace", []) + [{
                "step": step,
                "status": "completed",
                "duration_ms": elapsed,
                "intent": data.get("intent"),
                "reason": data.get("reason"),
            }],
        }

    except Exception as exc:
        log.warning("Supervisor LLM call failed: %s — defaulting to data_query", exc)
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "intent": "data_query",
            "requires_sql": True,
            "clarification_question": None,
            "current_step": step,
            "warnings": state.get("warnings", []) + [f"Supervisor warning: {exc}"],
            "trace": state.get("trace", []) + [{
                "step": step,
                "status": "warning",
                "duration_ms": elapsed,
                "error": str(exc),
            }],
        }
