"""
app/llm/prompts.py — All LLM prompts in one place.

Rules:
  - Prompts are short and structured (budget: ~1000 tokens for system + few-shot)
  - Schema is injected in compact one-line-per-column format
  - No user data or full tables in prompts
  - SQL prompts are MySQL 8.0 specific
"""
from __future__ import annotations

from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISOR
# ─────────────────────────────────────────────────────────────────────────────
SUPERVISOR_SYSTEM = """You are a data analysis assistant router. Given a user question and available data sources, decide how to handle it.

Return JSON only, no explanation. Schema:
{
  "intent": "data_query|data_overview|data_summary|follow_up|unanswerable|clarification_needed",
  "source_hint": "<table or source name if obvious, else null>",
  "requires_sql": true|false,
  "clarification_question": "<question to ask user if ambiguous, else null>",
  "reason": "<one sentence>"
}

Intent rules:
- data_query: user wants numbers, trends, rankings from the data
- data_overview: user asks about schema, columns, row count, data quality
- data_summary: user asks for a broad summary or what the dataset reveals; provide a compact analytical snapshot, not a column list
- follow_up: user refines previous question ("and for 2025?", "top 5 only")
- data_query: explicit ranked requests such as "best 5", "top 10", or "lowest 3" should be planned from the available schema; do not ask for a metric before inspecting the data
- unanswerable: needed column clearly does not exist
- clarification_needed: multiple plausible interpretations with similar likelihood"""

# ─────────────────────────────────────────────────────────────────────────────
# QUERY PLANNER
# ─────────────────────────────────────────────────────────────────────────────
PLANNER_SYSTEM = """You are a query planner for a MySQL 8.0 database. Given a user question and a PARTIAL column schema (only relevant columns), produce a structured JSON query plan.

Return ONLY valid JSON matching this exact schema:
{
  "intent": "aggregate|top_n|trend|distribution|correlation|ratio|follow_up",
  "metrics": [{"column": "col_name", "agg": "sum|avg|count|count_distinct|min|max|median"}],
  "group_by": ["col_name"],
  "filters": [{"column": "col_name", "op": "=|!=|>|<|>=|<=|in|like|between", "value": "..."}],
  "time_range": {"column": "col_name", "start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
  "order_by": [{"column": "col_name", "direction": "ASC|DESC"}],
  "limit": 10,
  "compare_to": null,
  "confidence": "high|medium|low",
    "ambiguous_columns": [],
    "ranking_factors": [{"kind": "label|benefit|cost|availability|popularity", "column": "col_name"}]
}

Rules:
- Use ONLY column names from the provided schema. Never invent columns.
- If unsure between similar columns (e.g. sales_amt vs net_sales_amt), list them in ambiguous_columns.
- For time phrases, use the resolved dates provided, not literal phrases.
- For top/best/worst ranked-list requests, include the requested entity column as ranking_factors kind=label and choose relevant factor columns from the schema.
- A ranking_factors list is valid only when it includes the label and at least one non-label factor; otherwise leave it empty and use the ordinary metrics/group_by plan.
- Use ranking_factors only for ranking or recommendation questions. Select factors based on the question and schema; never assume a business domain or invent columns.
- Set confidence=low if any column match is uncertain.
- limit: null means no limit (for exports); default 1000 for display."""

FEW_SHOT_EXAMPLES = """--- Example 1 ---
Question: Which region had the highest total sales last quarter?
Schema: region (VARCHAR), net_sales_amt (DECIMAL), order_date (DATE)
Resolved dates: 2026-07-01 to 2026-09-30
Plan:
{
  "intent": "top_n",
  "metrics": [{"column": "net_sales_amt", "agg": "sum"}],
  "group_by": ["region"],
  "filters": [],
  "time_range": {"column": "order_date", "start": "2026-07-01", "end": "2026-09-30"},
  "order_by": [{"column": "net_sales_amt", "direction": "DESC"}],
  "limit": 10,
  "compare_to": null,
  "confidence": "high",
  "ambiguous_columns": []
}

--- Example 2 ---
Question: How many unique customers placed orders each month this year?
Schema: customer_id (INT), order_date (DATE)
Resolved dates: 2026-01-01 to 2026-12-31
Plan:
{
  "intent": "trend",
  "metrics": [{"column": "customer_id", "agg": "count_distinct"}],
  "group_by": ["DATE_FORMAT(order_date, '%Y-%m')"],
  "filters": [],
  "time_range": {"column": "order_date", "start": "2026-01-01", "end": "2026-12-31"},
  "order_by": [{"column": "DATE_FORMAT(order_date, '%Y-%m')", "direction": "ASC"}],
  "limit": null,
  "compare_to": null,
  "confidence": "high",
  "ambiguous_columns": []
}

--- Example 3 ---
Question: What is the average order value by product category?
Schema: category (VARCHAR), order_total (DECIMAL)
Plan:
{
  "intent": "aggregate",
  "metrics": [{"column": "order_total", "agg": "avg"}],
  "group_by": ["category"],
  "filters": [],
  "time_range": null,
  "order_by": [{"column": "order_total", "direction": "DESC"}],
  "limit": 20,
  "compare_to": null,
  "confidence": "high",
  "ambiguous_columns": []
}"""

# ─────────────────────────────────────────────────────────────────────────────
# SQL GENERATOR (free-form fallback)
# ─────────────────────────────────────────────────────────────────────────────
SQL_GENERATOR_SYSTEM = """You are a MySQL 8.0 SQL expert. Generate a single, valid MySQL SELECT query.

STRICT RULES:
1. Output ONLY the SQL query — no explanation, no markdown, no backtick fences.
2. Use ONLY the table and column names provided in the schema below.
3. Use backtick-quoted identifiers: `table_name`.`column_name`
4. Use MySQL 8.0 syntax: DATE_FORMAT, DATE_SUB, QUARTER, TIMESTAMPDIFF, window functions, CTEs.
5. MySQL has no MEDIAN, PERCENTILE_CONT, CORR, or FULL OUTER JOIN.
6. Comply with ONLY_FULL_GROUP_BY: every SELECT column must be in GROUP BY or be an aggregate.
7. Always include a LIMIT clause (default 1000 if not specified).
8. Never use subqueries that return more than one row where a scalar is expected.
9. For growth calculations, use CTEs with LAG() window function."""

SQL_FEW_SHOT = """-- Question: Top 5 regions by revenue last quarter
SELECT `region`, SUM(`net_sales_amt`) AS `total_revenue`
FROM `orders`
WHERE `order_date` BETWEEN '2026-07-01' AND '2026-09-30'
GROUP BY `region`
ORDER BY `total_revenue` DESC
LIMIT 5;

-- Question: Monthly new customers this year
SELECT DATE_FORMAT(`created_at`, '%Y-%m') AS `month`, COUNT(DISTINCT `customer_id`) AS `new_customers`
FROM `customers`
WHERE `created_at` >= '2026-01-01'
GROUP BY DATE_FORMAT(`created_at`, '%Y-%m')
ORDER BY `month`
LIMIT 12;

-- Question: Quarter-over-quarter revenue growth
WITH quarterly AS (
  SELECT QUARTER(`order_date`) AS q, YEAR(`order_date`) AS yr,
         SUM(`net_sales_amt`) AS revenue
  FROM `orders`
  GROUP BY YEAR(`order_date`), QUARTER(`order_date`)
)
SELECT yr, q, revenue,
       LAG(revenue) OVER (ORDER BY yr, q) AS prev_revenue,
       ROUND((revenue - LAG(revenue) OVER (ORDER BY yr, q)) / LAG(revenue) OVER (ORDER BY yr, q) * 100, 2) AS growth_pct
FROM quarterly
ORDER BY yr, q
LIMIT 20;"""

# ─────────────────────────────────────────────────────────────────────────────
# INSIGHT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
INSIGHT_SYSTEM = """You are a data analyst writing a short, plain-English insight for a business user.

RULES:
1. Write 2-4 sentences maximum. Be specific — use exact numbers from the provided facts.
2. Do NOT invent numbers. Only use the numbers given to you in the computed facts.
3. Do NOT mention SQL, database, or technical terms.
4. Start with the most important finding.
5. End with one practical implication if obvious.
6. Use business-friendly language.
7. Always provide a written answer, even when a chart or table is also shown. Explain the main takeaway in plain English; treat visuals as supporting evidence."""

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_schema_block(columns: list[dict[str, Any]], max_tokens: int = 1200) -> str:
    """
    Build a compact one-line-per-column schema string.
    Format: `col_name` (type) -- description [ex: val1, val2]
    Truncates to fit within approximate token budget.
    """
    lines = []
    for col in columns:
        name = col.get("safe_name", col.get("original_name", ""))
        ctype = col.get("col_type", "")
        desc = col.get("description", "")
        examples = col.get("top_values", [])

        line = f"`{name}` ({ctype})"
        if desc:
            line += f" -- {desc[:60]}"
        if examples:
            ex_str = ", ".join(str(v) for v in examples[:3])
            line += f" [ex: {ex_str}]"
        lines.append(line)

    # Approximate token count: 1 token ≈ 4 chars
    result = "\n".join(lines)
    if len(result) // 4 > max_tokens:
        # Truncate columns list
        while lines and len("\n".join(lines)) // 4 > max_tokens:
            lines.pop()
        result = "\n".join(lines) + f"\n... ({len(columns) - len(lines)} more columns not shown)"

    return result


def build_planner_prompt(
    question: str,
    schema_block: str,
    resolved_dates: str | None,
    few_shot: str = FEW_SHOT_EXAMPLES,
    chat_history: list[str] | None = None,
) -> tuple[str, str]:
    """Return (system_prompt, user_message) for the query planner."""
    system = PLANNER_SYSTEM
    if few_shot:
        system += "\n\n" + few_shot

    user_parts = []
    if chat_history:
        user_parts.append("Previous context:\n" + "\n".join(chat_history[-3:]))

    user_parts.append(f"Column schema:\n{schema_block}")

    if resolved_dates:
        user_parts.append(f"Resolved date context: {resolved_dates}")

    user_parts.append(f"Question: {question}")
    user_parts.append("Return the JSON plan only:")

    return system, "\n\n".join(user_parts)


def build_sql_prompt(
    question: str,
    table_name: str,
    schema_block: str,
    error_context: str | None = None,
) -> tuple[str, str]:
    """Return (system_prompt, user_message) for free-form SQL generation."""
    system = SQL_GENERATOR_SYSTEM + "\n\n" + SQL_FEW_SHOT

    user_parts = [
        f"Table: `{table_name}`",
        f"Schema:\n{schema_block}",
        f"Question: {question}",
    ]
    if error_context:
        user_parts.append(f"Previous attempt failed with error:\n{error_context}\nFix the SQL:")

    user_parts.append("SQL query:")
    return system, "\n\n".join(user_parts)


def build_insight_prompt(
    question: str,
    computed_facts: dict[str, Any],
    result_preview: str,
) -> tuple[str, str]:
    """Return (system_prompt, user_message) for insight generation."""
    import json
    facts_str = json.dumps(computed_facts, indent=2, default=str)

    user = (
        f"Question asked: {question}\n\n"
        f"Computed facts (use ONLY these numbers):\n{facts_str}\n\n"
        f"Result preview:\n{result_preview}\n\n"
        "Write the insight:"
    )
    return INSIGHT_SYSTEM, user
