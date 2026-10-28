"""Build a concise, data-backed snapshot for broad dataset questions."""
from __future__ import annotations

import logging
import re
import time
from decimal import Decimal
from typing import Any

from sqlalchemy import text

from app.graph.state import AgentState

log = logging.getLogger(__name__)

_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")
_METRIC_HINTS = ("revenue", "sales", "cost", "price", "amount", "quantity", "count", "total",
                 "score", "rating", "profit", "salary", "income", "weight", "volume", "votes",
                 "value", "val", "measure", "metric", "expenditure", "surplus", "assets")
_DIMENSION_HINTS = ("category", "type", "status", "segment", "group", "class", "level", "tier",
                    "department", "region", "city", "country", "state", "channel", "source",
                    "gender", "grade", "brand", "currency")
_EXCLUDED_HINTS = ("address", "email", "phone", "latitude", "longitude", "postcode", "postal",
                   "description", "verbose", "url", "year", "yr", "period", "row_id", "_row_id")


def _quote(identifier: str) -> str:
    if not _IDENTIFIER.fullmatch(identifier):
        raise ValueError("Unsafe database identifier")
    return f"`{identifier}`"


def _display(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (float, Decimal)):
        number = float(value)
        if number.is_integer():
            return f"{int(number):,}"
        return f"{number:,.2f}"
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError, OverflowError):
        return str(value)


def _format_top_values(values: list[tuple[Any, Any]]) -> str:
    return ", ".join(f"{value} ({_display(count)})" for value, count in values if value is not None)


async def data_summary_node(state: AgentState) -> dict[str, Any]:
    """Summarize row coverage, useful measures and common categories without an LLM."""
    started = time.perf_counter()
    source = state.get("selected_source") or {}
    catalog = state.get("full_catalog", [])
    source_name = str(source.get("name") or "Dataset")
    row_count = source.get("row_count")
    col_count = source.get("col_count") or len(catalog)
    missing = [c for c in catalog if float(c.get("null_pct") or 0) > 20]

    try:
        from app.storage.database import readonly_session

        table = _quote(str(state.get("table_name") or source.get("table_name") or ""))
        schema = _quote(str(state.get("schema_name") or source.get("schema_name") or "quantai_data"))
        metrics = [
            c for c in catalog
            if c.get("col_type") in ("int", "float")
            and not c.get("is_key_col")
            and not c.get("is_pii")
            and not c.get("is_date_col")
            and c.get("unit") != "year"
            and not any(hint in c["safe_name"].lower() for hint in _EXCLUDED_HINTS)
        ]
        metrics.sort(
            key=lambda c: (
                not any(hint in c["safe_name"].lower() for hint in _METRIC_HINTS),
                c.get("ordinal_position", 0),
            )
        )
        metrics = metrics[:3]

        dimensions = [
            c for c in catalog
            if c.get("col_type") in ("str", "bool", "int")
            and not c.get("is_key_col")
            and not c.get("is_pii")
            and 1 < int(c.get("distinct_count") or 0) <= 1500
            and not any(hint in c["safe_name"].lower() for hint in _EXCLUDED_HINTS)
        ]
        dimensions.sort(
            key=lambda c: (
                not any(hint in c["safe_name"].lower() for hint in _DIMENSION_HINTS),
                int(c.get("distinct_count") or 0) > 100,
                c.get("ordinal_position", 0),
            )
        )
        dimensions = dimensions[:3]

        with readonly_session() as connection:
            if not row_count:
                row_count = connection.execute(text(f"SELECT COUNT(*) FROM {schema}.{table}")).scalar_one()

            numeric_facts: list[tuple[str, Any, Any, Any]] = []
            if metrics:
                expressions = []
                for index, column in enumerate(metrics):
                    name = _quote(column["safe_name"])
                    expressions.extend((
                        f"AVG({name}) AS `avg_{index}`",
                        f"MIN({name}) AS `min_{index}`",
                        f"MAX({name}) AS `max_{index}`",
                    ))
                row = connection.execute(text(
                    f"SELECT {', '.join(expressions)} FROM {schema}.{table}"
                )).mappings().one()
                for index, column in enumerate(metrics):
                    average = row.get(f"avg_{index}")
                    minimum = row.get(f"min_{index}")
                    maximum = row.get(f"max_{index}")
                    if average is not None:
                        numeric_facts.append((column["safe_name"], average, minimum, maximum))

            category_facts: list[tuple[str, list[tuple[Any, Any]]]] = []
            for column in dimensions:
                name = _quote(column["safe_name"])
                rows = connection.execute(text(
                    f"SELECT {name} AS value, COUNT(*) AS frequency "
                    f"FROM {schema}.{table} WHERE {name} IS NOT NULL "
                    f"GROUP BY {name} ORDER BY frequency DESC LIMIT 3"
                )).all()
                values = [(item[0], item[1]) for item in rows]
                if values:
                    category_facts.append((column["safe_name"], values))

    except Exception as exc:
        log.warning("Dataset profile query failed (%s); using catalog metadata", type(exc).__name__)
        numeric_facts = []
        category_facts = []

    lines = [
        f"## {source_name}: dataset snapshot",
        f"This dataset contains **{_display(row_count)} rows** across **{_display(col_count)} columns**.",
    ]

    if numeric_facts:
        lines.extend(("", "### Key measures"))
        for name, average, minimum, maximum in numeric_facts:
            lines.append(
                f"- **{name.replace('_', ' ').title()}** averages **{_display(average)}** "
                f"(range: {_display(minimum)} to {_display(maximum)})."
            )

    if category_facts:
        lines.extend(("", "### Most common groups"))
        for name, values in category_facts:
            lines.append(f"- **{name.replace('_', ' ').title()}:** {_format_top_values(values)}")

    if missing:
        labels = ", ".join(f"`{column['safe_name']}` ({_display(column.get('null_pct'))}% missing)" for column in missing[:4])
        lines.extend(("", f"**Data quality:** {len(missing)} column(s) have more than 20% missing values: {labels}."))
    else:
        lines.extend(("", "**Data quality:** no columns have more than 20% missing values."))

    date_columns = [c for c in catalog if c.get("is_date_col")]
    if date_columns:
        spans = [
            f"`{c['safe_name']}` ({c.get('col_min') or 'unknown'} to {c.get('col_max') or 'unknown'})"
            for c in date_columns[:3]
        ]
        lines.append("**Date coverage:** " + "; ".join(spans) + ".")

    if not numeric_facts and not category_facts:
        columns = ", ".join(f"`{c['safe_name']}`" for c in catalog[:6])
        lines.append(f"The available fields include {columns or 'no catalogued columns'}. Ask for a specific comparison or field to explore.")
    else:
        lines.extend(("", "Ask a focused follow-up to compare groups, explore a metric, or investigate a trend."))

    elapsed = int((time.perf_counter() - started) * 1000)
    return {
        "insight_text": "\n".join(lines),
        "generated_sql": None,
        "chart_json": None,
        "current_step": "data_summary",
        "trace": state.get("trace", []) + [{
            "step": "data_summary",
            "status": "completed",
            "duration_ms": elapsed,
            "measures": len(numeric_facts),
            "dimensions": len(category_facts),
        }],
    }
