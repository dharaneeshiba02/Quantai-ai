"""
app/agents/supervisor.py — Supervisor agent: routes questions and handles ambiguity.
"""
from __future__ import annotations

import re
import time
from typing import Any

from app.graph.state import AgentState

async def supervisor_node(state: AgentState) -> dict[str, Any]:
    """
    Route the user question to the appropriate intent.
    Returns state updates.
    """
    t0 = time.perf_counter()
    step = "supervisor"

    question = state["question"]
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
        "provide insight",
        "provide random insight",
        "random insight",
        "give me an insight",
        "give me a random insight",
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

    # Dataset questions are the default intent. The planner already understands
    # the question and decides how to query it, so a separate model call here
    # only adds latency without changing the route.
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
            "intent": "data_query",
            "note": "Default dataset intent; planner handles question interpretation",
        }],
    }
