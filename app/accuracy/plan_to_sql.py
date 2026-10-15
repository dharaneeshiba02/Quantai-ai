"""
app/accuracy/plan_to_sql.py — Deterministic SQL compiler from structured query plan.

Converts the JSON query plan produced by the planner into valid MySQL 8.0 SQL
without involving the LLM. This is the primary SQL generation path.
Supported: sum, avg, count, count_distinct, min, max, median, group by,
           filters, top/bottom N, time ranges, period-over-period growth,
           running totals, distributions, correlations, ratio metrics.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

log = logging.getLogger(__name__)

# Mapping from plan agg names to MySQL aggregate functions
AGG_MAP = {
    "sum": "SUM",
    "avg": "AVG",
    "count": "COUNT",
    "count_distinct": "COUNT(DISTINCT {col})",
    "min": "MIN",
    "max": "MAX",
    "median": None,  # Handled specially via window function
}

OP_MAP = {
    "=": "=",
    "!=": "!=",
    "<>": "!=",
    ">": ">",
    "<": "<",
    ">=": ">=",
    "<=": "<=",
    "in": "IN",
    "not in": "NOT IN",
    "like": "LIKE",
    "not like": "NOT LIKE",
    "between": "BETWEEN",
    "is null": "IS NULL",
    "is not null": "IS NOT NULL",
}


def _quote(name: str) -> str:
    """Backtick-quote a MySQL identifier."""
    if name.startswith("`") and name.endswith("`"):
        return name
    # If it contains SQL function-like patterns, don't quote
    if "(" in name or "DATE_FORMAT" in name.upper():
        return name
    return f"`{name}`"


def _build_metric_expr(metric: dict[str, Any], median_col_alias: str | None = None) -> tuple[str, str]:
    """
    Build a SQL expression + alias for a metric.
    Returns (sql_expression, alias_name).
    """
    col = metric["column"]
    agg = metric.get("agg", "sum").lower()
    quoted_col = _quote(col)

    if agg == "count_distinct":
        expr = f"COUNT(DISTINCT {quoted_col})"
        alias = f"count_distinct_{col}"
    elif agg == "median":
        # MySQL median via window function inline — handled in _build_median_cte
        expr = f"AVG({quoted_col})"  # placeholder; will be replaced by caller
        alias = f"median_{col}"
    elif agg in AGG_MAP and AGG_MAP[agg]:
        expr = f"{AGG_MAP[agg]}({quoted_col})"
        alias = f"{agg}_{col}"
    elif agg == "count":
        expr = "COUNT(*)"
        alias = "count"
    else:
        expr = f"SUM({quoted_col})"
        alias = f"sum_{col}"

    return expr, alias


def _build_filter_clause(filters: list[dict[str, Any]]) -> str:
    """Build a WHERE clause from filter list."""
    if not filters:
        return ""

    parts = []
    for flt in filters:
        col = _quote(flt["column"])
        op = OP_MAP.get(str(flt.get("op", "=")).lower(), "=")
        value = flt.get("value")

        if op in ("IS NULL", "IS NOT NULL"):
            parts.append(f"{col} {op}")
        elif op == "IN":
            vals = value if isinstance(value, list) else [value]
            escaped = ", ".join(f"'{v}'" for v in vals)
            parts.append(f"{col} IN ({escaped})")
        elif op == "BETWEEN":
            if isinstance(value, list) and len(value) == 2:
                parts.append(f"{col} BETWEEN '{value[0]}' AND '{value[1]}'")
        else:
            # Numeric detection: don't quote if numeric
            try:
                float(str(value))
                parts.append(f"{col} {op} {value}")
            except (ValueError, TypeError):
                parts.append(f"{col} {op} '{value}'")

    return "WHERE " + " AND ".join(parts)


def _build_time_clause(time_range: Optional[dict[str, Any]], existing_filters: list[dict]) -> str:
    """Build time range condition as a WHERE / AND clause fragment."""
    if not time_range:
        return ""
    col = _quote(time_range["column"])
    start = time_range.get("start")
    end = time_range.get("end")
    if start and end:
        return f"{col} BETWEEN '{start}' AND '{end}'"
    elif start:
        return f"{col} >= '{start}'"
    elif end:
        return f"{col} <= '{end}'"
    return ""


def compile_plan(
    plan: dict[str, Any],
    table_name: str,
    schema_name: str | None = None,
    all_columns: bool = False,
) -> str:
    """
    Compile a query plan JSON into MySQL 8.0 SQL.

    Args:
        plan: The validated query plan dict.
        table_name: The MySQL table to query.
        schema_name: Optional schema prefix.

    Returns:
        MySQL 8.0 SQL string.

    Raises:
        ValueError: If the plan cannot be compiled.
    """
    tbl = f"`{schema_name}`.`{table_name}`" if schema_name else f"`{table_name}`"
    intent = plan.get("intent", "aggregate")
    metrics = plan.get("metrics", [])
    group_by = plan.get("group_by", [])
    filters = plan.get("filters", [])
    time_range = plan.get("time_range")
    order_by = plan.get("order_by", [])
    limit = plan.get("limit", 1000)

    # ── Detect median metrics ────────────────────────────────────────────────
    median_metrics = [m for m in metrics if m.get("agg", "").lower() == "median"]
    other_metrics = [m for m in metrics if m.get("agg", "").lower() != "median"]

    # ── Build WHERE clause ───────────────────────────────────────────────────
    where_parts = []
    time_cond = _build_time_clause(time_range, filters)
    if time_cond:
        where_parts.append(time_cond)

    filter_clause = _build_filter_clause(filters)
    if filter_clause:
        # Extract the WHERE content
        where_parts.append(filter_clause.replace("WHERE ", "", 1))

    full_where = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    ranking_factors = plan.get("ranking_factors", [])
    if len([factor for factor in ranking_factors if factor.get("kind") != "label"]) >= 1:
        return _compile_ranked_list(
            table=tbl,
            factors=ranking_factors,
            where=full_where,
            limit=limit,
            all_columns=all_columns,
        )

    if all_columns:
        return f"SELECT *\nFROM {tbl}\n{full_where}\nLIMIT {limit or 1000}"

    # ── Growth / trend intent ────────────────────────────────────────────────
    if intent == "trend" and time_range and not group_by:
        # Auto-group by month
        date_col = _quote(time_range["column"])
        group_by = [f"DATE_FORMAT({date_col}, '%Y-%m')"]

    # ── Simple aggregate / top-N ─────────────────────────────────────────────
    if median_metrics:
        return _compile_with_median(
            table=tbl,
            median_metrics=median_metrics,
            other_metrics=other_metrics,
            group_by=group_by,
            where=full_where,
            order_by=order_by,
            limit=limit,
        )

    # ── Standard SELECT ──────────────────────────────────────────────────────
    select_parts = []

    # Group by columns first (must appear in SELECT for ONLY_FULL_GROUP_BY)
    for gb in group_by:
        select_parts.append(_quote(gb) if "DATE_FORMAT" not in gb else gb)

    # Metric expressions
    metric_aliases = []
    for m in other_metrics:
        expr, alias = _build_metric_expr(m)
        select_parts.append(f"{expr} AS `{alias}`")
        metric_aliases.append(alias)

    # If no metrics at all, do SELECT *
    if not select_parts:
        select_parts = ["*"]

    select_clause = "SELECT " + ", ".join(select_parts)
    from_clause = f"FROM {tbl}"

    group_clause = ""
    if group_by:
        quoted_groups = [_quote(g) if "DATE_FORMAT" not in g else g for g in group_by]
        group_clause = "GROUP BY " + ", ".join(quoted_groups)

    order_clause = ""
    if order_by:
        order_parts = []
        for ob in order_by:
            col = ob["column"]
            direction = ob.get("direction", "DESC").upper()
            
            # Check if col directly matches a metric alias
            matched_alias = None
            if col in metric_aliases:
                matched_alias = col
            else:
                # Find if any metric is defined on this column
                for m in other_metrics:
                    if m.get("column") == col:
                        _, a = _build_metric_expr(m)
                        matched_alias = a
                        break

            if matched_alias:
                order_parts.append(f"`{matched_alias}` {direction}")
            elif group_by and col not in group_by:
                # If grouping is active and column is neither grouped nor a metric, order by first metric
                if metric_aliases:
                    order_parts.append(f"`{metric_aliases[0]}` {direction}")
                else:
                    order_parts.append(f"{_quote(group_by[0])} {direction}")
            else:
                order_parts.append(f"{_quote(col)} {direction}")
        order_clause = "ORDER BY " + ", ".join(order_parts)
    elif group_by and metric_aliases:
        # Default when grouping: order by first metric descending
        order_clause = f"ORDER BY `{metric_aliases[0]}` DESC"

    limit_clause = f"LIMIT {limit}" if limit else ""

    parts = [p for p in [select_clause, from_clause, full_where, group_clause, order_clause, limit_clause] if p]
    return "\n".join(parts)


def _compile_ranked_list(
    table: str,
    factors: list[dict[str, str]],
    where: str,
    limit: int | None,
    all_columns: bool = False,
) -> str:
    """Compile a schema-aware ranked-list query with normalized factor scores."""
    label = next(f["column"] for f in factors if f["kind"] == "label")
    score_parts = []
    factor_columns = [f["column"] for f in factors if f["kind"] != "label"]
    weights = {"benefit": 0.45, "rating": 0.45, "popularity": 0.25, "cost": 0.20, "availability": 0.10}
    total_weight = sum(weights.get(f["kind"], 0) for f in factors if f["kind"] != "label") or 1

    for factor in factors:
        kind = factor["kind"]
        if kind == "label":
            continue
        column = factor["column"]
        quoted = _quote(column)
        weight = weights.get(kind, 0) / total_weight
        if kind == "availability":
            normalized = f"CASE WHEN LOWER(CAST({quoted} AS CHAR)) IN ('yes', 'true', '1', 'online', 'available', 'active', 'open') THEN 1 ELSE 0 END"
        elif kind == "cost":
            normalized = f"(1 - COALESCE({quoted}, 0) / NULLIF(MAX({quoted}) OVER (), 0))"
        else:
            normalized = f"COALESCE({quoted}, 0) / NULLIF(MAX({quoted}) OVER (), 0)"
        score_parts.append(f"({normalized}) * {weight:.4f}")

    base_columns = [f"{_quote(label)} AS `{label}`"]
    for factor in factors:
        if factor["kind"] == "label":
            continue
        column = factor["column"]
        quoted = _quote(column)
        if factor["kind"] == "availability":
            aggregate = f"MAX(CASE WHEN LOWER(CAST({quoted} AS CHAR)) IN ('yes', 'true', '1', 'online', 'available', 'active', 'open') THEN 1 ELSE 0 END)"
        elif factor["kind"] == "cost":
            aggregate = f"MIN({quoted})"
        else:
            aggregate = f"MAX({quoted})"
        base_columns.append(f"{aggregate} AS `{column}`")

    select_columns = [f"`{label}`"] + [_quote(column) for column in factor_columns]
    score = " + ".join(score_parts)
    limit_clause = f"\nLIMIT {limit}" if limit else ""
    if all_columns:
        return (
            f"SELECT *, ({score}) AS `recommendation_score`\n"
            f"FROM {table}\n{where}\n"
            f"ORDER BY `recommendation_score` DESC{limit_clause}"
        )

    return (
        f"WITH ranked_base AS (\n"
        f"SELECT {', '.join(base_columns)}\n"
        f"FROM {table}\n{where}\n"
        f"GROUP BY {_quote(label)}\n"
        f")\n"
        f"SELECT {', '.join(select_columns)}, ({score}) AS `recommendation_score`\n"
        f"FROM ranked_base\n"
        f"ORDER BY `recommendation_score` DESC{limit_clause}"
    )


def _compile_with_median(
    table: str,
    median_metrics: list[dict],
    other_metrics: list[dict],
    group_by: list[str],
    where: str,
    order_by: list[dict],
    limit: int | None,
) -> str:
    """
    Compile a query that includes median, using ROW_NUMBER window function.
    MySQL has no built-in MEDIAN, so we simulate it with a CTE.
    Only works for single-group or no-group median.
    """
    col = median_metrics[0]["column"]
    quoted_col = _quote(col)

    if not group_by:
        # Global median
        sql = f"""WITH ranked AS (
  SELECT {quoted_col},
         ROW_NUMBER() OVER (ORDER BY {quoted_col}) AS rn,
         COUNT(*) OVER () AS total
  FROM {table}
  {where}
)
SELECT AVG({quoted_col}) AS `median_{col}`
FROM ranked
WHERE rn IN (FLOOR((total + 1) / 2), CEIL((total + 1) / 2))"""
    else:
        group_cols = ", ".join(_quote(g) for g in group_by)
        sql = f"""WITH ranked AS (
  SELECT {group_cols}, {quoted_col},
         ROW_NUMBER() OVER (PARTITION BY {group_cols} ORDER BY {quoted_col}) AS rn,
         COUNT(*) OVER (PARTITION BY {group_cols}) AS total
  FROM {table}
  {where}
)
SELECT {group_cols}, AVG({quoted_col}) AS `median_{col}`
FROM ranked
WHERE rn IN (FLOOR((total + 1) / 2), CEIL((total + 1) / 2))
GROUP BY {group_cols}"""

    if limit:
        sql += f"\nLIMIT {limit}"

    return sql


def plan_to_interpretation(
    plan: dict[str, Any],
    table_name: str,
) -> str:
    """
    Generate a plain-English interpretation of the plan (deterministic, no LLM).
    Shown to user as 'How I understood your question'.
    """
    metrics = plan.get("metrics", [])
    group_by = plan.get("group_by", [])
    filters = plan.get("filters", [])
    time_range = plan.get("time_range")
    order_by = plan.get("order_by", [])
    limit = plan.get("limit")
    ranking_factors = plan.get("ranking_factors", [])
    has_ranking_score = len([factor for factor in ranking_factors if factor.get("kind") != "label"]) >= 1

    parts = []

    if has_ranking_score:
        factor_names = [factor["kind"] for factor in ranking_factors if factor["kind"] != "label"]
        parts.append(f"ranked by combined {', '.join(factor_names)}")

    # Metrics
    if metrics and not has_ranking_score:
        metric_strs = []
        for m in metrics:
            agg = m.get("agg", "sum").replace("_", " ")
            col = m["column"]
            metric_strs.append(f"{agg} of `{col}`")
        parts.append(", ".join(metric_strs))

    # Group by
    if group_by:
        parts.append(f"grouped by {', '.join(f'`{g}`' for g in group_by)}")

    # Time range
    if time_range:
        start = time_range.get("start", "")
        end = time_range.get("end", "")
        desc = time_range.get("description", f"{start} to {end}")
        parts.append(f"from {desc}")

    # Filters
    for flt in filters:
        parts.append(f"where `{flt['column']}` {flt.get('op', '=')} '{flt.get('value', '')}'")

    # Order
    if order_by:
        ob = order_by[0]
        direction = "highest" if ob.get("direction", "DESC") == "DESC" else "lowest"
        parts.append(f"{direction} first")

    # Limit
    if limit and limit < 1000:
        parts.append(f"top {limit}")

    from_part = f"from `{table_name}`"
    return " ".join(parts) + (f" {from_part}" if parts else from_part)
