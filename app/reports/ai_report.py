"""Local-Qwen report reasoning and chart-selection enrichment."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from copy import deepcopy
from typing import Any

from app.config import get_settings

log = logging.getLogger(__name__)

REPORT_REASONING_SYSTEM = """You are QuantAI's senior data visualization and reporting analyst.

Review the deterministic dataset profile and candidate charts supplied by the application. Choose only charts that directly answer a useful analytical question for this dataset. Do not choose a chart just because it is available.

Chart selection rules:
- Use a line chart only for ordered time or sequence data.
- Use a bar chart for category comparisons or rankings.
- Use a histogram for the distribution of one continuous numeric measure.
- Use a heatmap only when there are at least two meaningful numeric measures and the user needs relationships.
- Use a missing-values chart only when data quality is relevant.
- Avoid duplicate charts, charts based on identifiers, and charts with too many categories to read.
- Prefer at most three charts and explain what each selected chart is intended to reveal.

Reasoning rules:
- Use only values in the profile. Never invent numbers.
- Distinguish observed evidence from interpretation and do not claim causation.
- Explain the most important limitation, such as missing values, sampling, small groups, or mixed units.
- Do not expose private chain-of-thought. Return only valid JSON matching the schema.

JSON schema:
{
  "selected_chart_titles": ["exact candidate title"],
  "executive_summary": "2-4 concise sentences grounded in the profile",
  "key_findings": ["specific finding with exact values"],
  "chart_explanations": {"exact candidate title": "what the selected chart explains"},
  "caveats": ["important limitation or follow-up"]
}"""


async def enrich_dataset_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Ask local Qwen to select and explain report charts, with safe deterministic fallback."""
    cfg = get_settings()
    enriched = deepcopy(profile)
    enriched["ai_report_status"] = "deterministic"
    enriched["ai_report_model"] = cfg.llm_model

    if not cfg.llm_report_ai_enabled:
        return enriched

    candidates = [
        {
            "title": chart.get("title"),
            "kind": chart.get("kind"),
            "caption": chart.get("caption"),
            "interpretation": chart.get("interpretation"),
            "labels": chart.get("labels", [])[:8],
            "values": chart.get("values", [])[:8],
            "group_counts": chart.get("group_counts", [])[:8],
        }
        for chart in profile.get("charts", [])
    ]
    if not candidates:
        enriched["ai_report_status"] = "deterministic_no_charts"
        return enriched

    user_message = (
        "Dataset profile:\n"
        + json.dumps(_profile_for_prompt(profile), indent=2, default=str)
        + "\n\nCandidate charts:\n"
        + json.dumps(candidates, indent=2, default=str)
        + "\n\nSelect the most useful charts and write the report reasoning JSON."
    )

    try:
        from app.llm.client import call_llm

        log.info("Generating local Qwen report reasoning with model %s", cfg.llm_model)
        response = await asyncio.wait_for(
            call_llm(
                REPORT_REASONING_SYSTEM,
                user_message,
                temperature=0.0,
                format={"type": "json_object"},
            ),
            timeout=cfg.llm_report_timeout_sec,
        )
        result = _parse_json(response)
        if not isinstance(result, dict):
            raise ValueError("Report reasoning response was not an object")
        _apply_model_result(enriched, result)
        enriched["ai_report_status"] = "qwen3.5"
        return enriched
    except Exception as exc:
        log.warning("Local Qwen report reasoning unavailable; using deterministic report: %s", type(exc).__name__)
        enriched["ai_report_status"] = f"deterministic_fallback:{type(exc).__name__}"
        return enriched


def _profile_for_prompt(profile: dict[str, Any]) -> dict[str, Any]:
    """Keep the report prompt bounded and exclude file paths or sensitive catalog data."""
    return {
        "name": profile.get("name"),
        "row_count": profile.get("row_count"),
        "col_count": profile.get("col_count"),
        "summary": profile.get("summary"),
        "quality_notes": profile.get("quality_notes", [])[:8],
        "measures": profile.get("measures", [])[:8],
        "distribution_summary": profile.get("distribution_summary"),
        "categories": profile.get("categories", [])[:6],
    }


def _parse_json(value: str) -> Any:
    text = str(value or "").strip()
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def _apply_model_result(profile: dict[str, Any], result: dict[str, Any]) -> None:
    charts = profile.get("charts", [])
    by_title = {str(chart.get("title")): chart for chart in charts}
    requested_titles = [str(title) for title in result.get("selected_chart_titles", [])]
    selected = [by_title[title] for title in requested_titles if title in by_title]
    if selected:
        profile["charts"] = selected[:3]
        selected_titles = {str(chart.get("title")) for chart in selected}
        profile["chart_summaries"] = [
            item for item in profile.get("chart_summaries", []) if item.get("title") in selected_titles
        ]

    summary = str(result.get("executive_summary") or "").strip()
    if summary:
        profile["summary"] = summary

    findings = [str(item).strip() for item in result.get("key_findings", []) if str(item).strip()]
    if findings:
        profile["highlights"] = findings[:6]

    explanations = result.get("chart_explanations") or {}
    if isinstance(explanations, dict):
        for chart in profile.get("charts", []):
            explanation = str(explanations.get(chart.get("title"), "")).strip()
            if explanation:
                chart["ai_description"] = explanation

    caveats = [str(item).strip() for item in result.get("caveats", []) if str(item).strip()]
    if caveats:
        profile["quality_notes"] = list(dict.fromkeys(profile.get("quality_notes", []) + caveats))[:10]
