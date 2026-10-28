"""
app/agents/visualizer.py — Visualization agent (deterministic chart type selection).
Generates Plotly figure JSON based on result shape and question semantics — no LLM involved.

Supported chart types:
  - bar (vertical & horizontal)
  - grouped_bar / stacked_bar
  - line & area
  - scatter & bubble
  - histogram & box plot
  - pie / donut
  - correlation heatmap
"""
from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from app.graph.state import AgentState, cache_get

log = logging.getLogger(__name__)


def _select_chart_type(df: pd.DataFrame, stats: dict, question: str = "") -> str:
    """
    Deterministically select the best chart type from result shape and intent.
    """
    q = question.lower()
    cols = list(df.columns)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    date_cols = [c for c in cols if any(
        kw in c.lower() for kw in ["date", "month", "year", "quarter", "week", "period", "time", "day"]
    )]

    n_num = len(numeric)
    n_cat = len(text_cols)
    n_rows = len(df)

    if n_num == 0:
        return "table"

    # Specific question intent overrides
    if any(kw in q for kw in ["box", "boxplot", "spread", "quartile", "iqr"]):
        return "box"

    if any(kw in q for kw in ["heatmap", "correlat"]) and n_num >= 2:
        return "heatmap"

    if any(kw in q for kw in ["area", "cumulative", "stack"]) and (date_cols or n_cat >= 1):
        return "area"

    if any(kw in q for kw in ["share", "proportion", "breakdown", "distribution", "percent", "pie"]):
        if text_cols and n_num and df[text_cols[0]].nunique() <= 8:
            return "pie"

    # Date / Time-series -> Line or Area
    if n_num == 1 and (date_cols or (n_cat == 1 and n_rows >= 6)):
        if "area" in q or "volume" in q:
            return "area"
        return "line"

    # Single categorical + single numeric
    if n_num == 1 and n_cat >= 1:
        # Check label length: long labels look much better in horizontal bars
        first_cat = text_cols[0]
        max_label_len = df[first_cat].astype(str).str.len().max() if len(df) > 0 else 0
        if max_label_len > 16 or any(kw in q for kw in ["top", "rank", "lowest", "bottom"]):
            return "horizontal_bar"
        return "bar"

    # Multiple numeric columns without category -> correlation or scatter
    if n_num >= 2 and n_cat == 0:
        if n_num > 2 or "heatmap" in q or "matrix" in q:
            return "heatmap"
        return "scatter"

    # Multiple numeric columns with categories -> grouped bar
    if n_num >= 2 and n_cat >= 1:
        return "grouped_bar"

    # Single numeric column -> histogram or box
    if n_num == 1 and n_cat == 0:
        return "histogram"

    return "bar"


async def visualizer_node(state: AgentState) -> dict[str, Any]:
    """Generate a Plotly chart from the query result."""
    t0 = time.perf_counter()
    step = "visualizer"
    run_id = state["run_id"]

    try:
        df: pd.DataFrame | None = cache_get(run_id, "result_df")
        if df is None or df.empty:
            elapsed = int((time.perf_counter() - t0) * 1000)
            return {
                "chart_json": None,
                "chart_type": None,
                "current_step": step,
                "trace": state.get("trace", []) + [{"step": step, "status": "skipped_empty", "duration_ms": elapsed}],
            }

        stats = state.get("stats") or {}
        question = state.get("question", "")

        chart_type = _select_chart_type(df, stats, question)

        fig = _build_chart(df, chart_type, state)

        # Apply consistent styling
        fig = _apply_theme(fig)

        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "chart_json": fig.to_dict(),
            "chart_type": chart_type,
            "current_step": step,
            "trace": state.get("trace", []) + [{
                "step": step,
                "status": "completed",
                "duration_ms": elapsed,
                "chart_type": chart_type,
            }],
        }

    except Exception as exc:
        log.exception("Visualizer error")
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "chart_json": None,
            "chart_type": None,
            "current_step": step,
            "warnings": state.get("warnings", []) + [f"Chart generation failed: {exc}"],
            "trace": state.get("trace", []) + [{"step": step, "status": "warning", "duration_ms": elapsed}],
        }


def _build_chart(df: pd.DataFrame, chart_type: str, state: AgentState) -> go.Figure:
    """Build a Plotly figure for the given chart type."""
    question = state.get("question", "")

    cols = list(df.columns)
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    text_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Preview limited rows for chart performance
    chart_df = df.head(100)

    title_text = question[:80] if question else "Analysis Result"

    if chart_type == "bar":
        x_col = text_cols[0] if text_cols else cols[0]
        y_col = numeric[-1] if numeric else cols[-1]
        fig = px.bar(chart_df, x=x_col, y=y_col, title=title_text)

    elif chart_type == "horizontal_bar":
        cat_col = text_cols[0] if text_cols else cols[0]
        num_col = numeric[-1] if numeric else cols[-1]
        # Sort ascending so top rank is at the top of horizontal chart
        sorted_df = chart_df.sort_values(num_col, ascending=True)
        fig = px.bar(sorted_df, x=num_col, y=cat_col, orientation="h", title=title_text)

    elif chart_type == "line":
        date_cols = [c for c in cols if any(
            kw in c.lower() for kw in ["date", "month", "year", "quarter", "week", "period", "time", "day"]
        )]
        x_col = date_cols[0] if date_cols else (text_cols[0] if text_cols else cols[0])
        y_col = numeric[-1] if numeric else cols[-1]
        sorted_df = chart_df.sort_values(x_col)
        fig = px.line(sorted_df, x=x_col, y=y_col, title=title_text, markers=True)

    elif chart_type == "area":
        date_cols = [c for c in cols if any(
            kw in c.lower() for kw in ["date", "month", "year", "quarter", "week", "period", "time", "day"]
        )]
        x_col = date_cols[0] if date_cols else (text_cols[0] if text_cols else cols[0])
        y_col = numeric[-1] if numeric else cols[-1]
        sorted_df = chart_df.sort_values(x_col)
        fig = px.area(sorted_df, x=x_col, y=y_col, title=title_text)

    elif chart_type == "pie":
        name_col = text_cols[0] if text_cols else cols[0]
        val_col = numeric[-1] if numeric else cols[-1]
        fig = px.pie(chart_df, names=name_col, values=val_col, title=title_text, hole=0.35)

    elif chart_type == "scatter":
        x_col = numeric[0] if len(numeric) >= 2 else cols[0]
        y_col = numeric[1] if len(numeric) >= 2 else cols[1]
        color_col = text_cols[0] if text_cols else None
        fig = px.scatter(chart_df, x=x_col, y=y_col, color=color_col, title=title_text)

    elif chart_type == "box":
        y_col = numeric[0] if numeric else cols[0]
        x_col = text_cols[0] if text_cols else None
        fig = px.box(chart_df, x=x_col, y=y_col, title=title_text, points="outliers")

    elif chart_type == "histogram":
        col = numeric[0] if numeric else cols[0]
        fig = px.histogram(chart_df, x=col, title=title_text, marginal="box")

    elif chart_type == "heatmap":
        if len(numeric) >= 2:
            corr = chart_df[numeric].corr().round(2)
            fig = px.imshow(
                corr,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Viridis",
                title=f"Correlation Heatmap — {title_text}",
            )
        else:
            fig = px.bar(chart_df, x=cols[0], y=cols[-1], title=title_text)

    elif chart_type == "grouped_bar":
        x_col = text_cols[0] if text_cols else cols[0]
        fig = go.Figure()
        for num_col in numeric[:5]:
            fig.add_trace(go.Bar(name=num_col, x=chart_df[x_col], y=chart_df[num_col]))
        fig.update_layout(barmode="group", title=title_text)

    else:
        # Fallback
        x_col = cols[0]
        y_col = numeric[0] if numeric else cols[min(1, len(cols) - 1)]
        fig = px.bar(chart_df, x=x_col, y=y_col, title=title_text)

    return fig


def _apply_theme(fig: go.Figure) -> go.Figure:
    """Apply consistent QuantAI theme to the chart."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter, system-ui, sans-serif", "size": 13},
        margin={"l": 40, "r": 20, "t": 50, "b": 40},
        legend={"orientation": "h", "y": -0.15},
        colorway=["#6366f1", "#8b5cf6", "#06b6d4", "#10b981", "#f59e0b", "#ef4444"],
        xaxis={"showgrid": False, "zeroline": False},
        yaxis={"showgrid": True, "gridcolor": "rgba(100,100,100,0.15)", "zeroline": False},
        hovermode="x unified",
    )
    return fig
