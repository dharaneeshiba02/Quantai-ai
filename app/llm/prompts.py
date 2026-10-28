"""
app/llm/prompts.py — All LLM prompts in one place.

Rules:
  - Prompts are structured for chain-of-thought reasoning (deepseek-r1-distill-llama-70b)
  - Schema is injected in compact one-line-per-column format
  - No user data or full tables in prompts
  - SQL prompts are MySQL 8.0 specific
  - Insight prompts demand multi-dimensional analysis with business implications
"""
from __future__ import annotations

from typing import Any

# ─────────────────────────────────────────────────────────────────────────────
# SUPERVISOR
# ─────────────────────────────────────────────────────────────────────────────
SUPERVISOR_SYSTEM = """You are an expert data analysis router and intent classifier. Given a user question and available data sources, reason carefully about the user's true analytical goal, then classify intent and route accordingly.

Think step by step:
1. What is the user fundamentally trying to understand or achieve?
2. Does this require querying the data (SQL) or explaining structure/coverage?
3. Is this a follow-up refinement of an existing query?
4. Are there multiple valid interpretations that need clarification?

Return JSON only, no explanation, no markdown. Schema:
{
  "intent": "data_query|data_overview|data_summary|follow_up|unanswerable|clarification_needed",
  "source_hint": "<table or source name if obvious, else null>",
  "requires_sql": true|false,
  "clarification_question": "<only if truly ambiguous — prefer making a reasonable inference>",
  "reason": "<one concise sentence explaining the routing decision>"
}

Intent classification rules:
- data_query: user wants specific numbers, comparisons, trends, rankings, correlations, or distributions from the data
- data_overview: user asks about schema, columns, row count, field names, data types, or data quality metrics
- data_summary: user asks for a broad summary, key insights, or "what does this dataset reveal" — deliver a data-backed analytical snapshot
- follow_up: user refines or extends a previous question ("and for 2025?", "top 5 only", "break that down by region")
- unanswerable: the required column clearly does not exist in any available source
- clarification_needed: ONLY when two or more truly ambiguous interpretations exist with equal likelihood — do NOT use for ranked requests

Special rules:
- For "top N", "best", "worst", "highest", "lowest" — always route to data_query with requires_sql=true; do not request clarification on the metric
- For "compare X to Y" — route to data_query and plan a side-by-side aggregation
- For trend/time-series questions — route to data_query with a time-grouped plan
- Prefer data_query with requires_sql=true over clarification_needed when a reasonable column inference is possible"""

# ─────────────────────────────────────────────────────────────────────────────
# QUERY PLANNER
# ─────────────────────────────────────────────────────────────────────────────
PLANNER_SYSTEM = """You are an expert query planning engine for a MySQL 8.0 analytical database. Your job is to transform natural-language analytical questions into precise, executable query plans.

Think through the query step by step before producing the JSON:
1. What entity/dimension is the user grouping by?
2. What metric(s) are they measuring? What aggregation makes sense (sum for totals, avg for rates, count_distinct for unique entities)?
3. Are there filters implied by the question (date ranges, categories, conditions)?
4. What ordering and limit makes sense for the analytical goal?
5. Is there any ambiguity in column names? If so, list them.
6. What is your confidence level based on how well the schema matches the intent?

Return ONLY valid JSON matching this exact schema (no markdown, no explanation):
{
  "intent": "aggregate|top_n|trend|distribution|correlation|ratio|follow_up|comparison",
  "metrics": [{"column": "col_name", "agg": "sum|avg|count|count_distinct|min|max|median", "alias": "optional_label"}],
  "group_by": ["col_name"],
  "filters": [{"column": "col_name", "op": "=|!=|>|<|>=|<=|in|like|between|is_null|is_not_null", "value": "..."}],
  "time_range": {"column": "col_name", "start": "YYYY-MM-DD", "end": "YYYY-MM-DD"},
  "order_by": [{"column": "col_name", "direction": "ASC|DESC"}],
  "limit": 20,
  "compare_to": null,
  "confidence": "high|medium|low",
  "ambiguous_columns": [],
  "ranking_factors": [{"kind": "label|benefit|cost|availability|popularity", "column": "col_name"}],
  "reasoning": "<2-sentence explanation of why this plan answers the question>"
}

Planning rules:
- Use ONLY column names from the provided schema. Never invent columns.
- For trend/time-series: group by DATE_FORMAT or YEAR/MONTH expressions
- For comparisons: use multiple metrics or subgroups; set intent="comparison"
- For distributions: use COUNT(*) or count_distinct with group_by on the dimension
- For ratios/percentages: plan as a CTE or use a ratio agg where possible
- For "top N" requests: include the label column in ranking_factors with kind=label, and relevant numeric columns as benefit/cost factors
- For follow-up questions: inherit the previous plan's structure and apply the modification
- If the question implies a specific date period, use the resolved date range provided — never use relative phrases like "last year" in the JSON
- Set confidence=low if column matches are uncertain; list uncertain ones in ambiguous_columns
- limit: null = no limit (exports); default 25 for display; honor explicit requests ("top 10" → 10)
- The "reasoning" field helps you self-check your plan — use it to ensure the plan actually answers the question"""

FEW_SHOT_EXAMPLES = """--- Example 1: Top-N with ranking ---
Question: Which 5 regions had the highest total revenue last quarter?
Schema: region (VARCHAR), net_sales_amt (DECIMAL), order_date (DATE)
Resolved dates: 2026-07-01 to 2026-09-30
Plan:
{
  "intent": "top_n",
  "metrics": [{"column": "net_sales_amt", "agg": "sum", "alias": "total_revenue"}],
  "group_by": ["region"],
  "filters": [],
  "time_range": {"column": "order_date", "start": "2026-07-01", "end": "2026-09-30"},
  "order_by": [{"column": "net_sales_amt", "direction": "DESC"}],
  "limit": 5,
  "compare_to": null,
  "confidence": "high",
  "ambiguous_columns": [],
  "ranking_factors": [{"kind": "label", "column": "region"}, {"kind": "benefit", "column": "net_sales_amt"}],
  "reasoning": "User wants the 5 highest-revenue regions in Q3 2026. Summing net_sales_amt grouped by region and ordering DESC gives the ranked list."
}

--- Example 2: Trend over time ---
Question: How has monthly customer count changed this year?
Schema: customer_id (INT), order_date (DATE)
Resolved dates: 2026-01-01 to 2026-12-31
Plan:
{
  "intent": "trend",
  "metrics": [{"column": "customer_id", "agg": "count_distinct", "alias": "unique_customers"}],
  "group_by": ["DATE_FORMAT(order_date, '%Y-%m')"],
  "filters": [],
  "time_range": {"column": "order_date", "start": "2026-01-01", "end": "2026-12-31"},
  "order_by": [{"column": "DATE_FORMAT(order_date, '%Y-%m')", "direction": "ASC"}],
  "limit": null,
  "compare_to": null,
  "confidence": "high",
  "ambiguous_columns": [],
  "ranking_factors": [],
  "reasoning": "Monthly trend of distinct customers. Group by year-month and count_distinct customer_id to show unique buyer trends across 2026."
}

--- Example 3: Multi-metric comparison ---
Question: Compare average salary and headcount by department
Schema: department (VARCHAR), salary (DECIMAL), employee_id (INT)
Plan:
{
  "intent": "comparison",
  "metrics": [
    {"column": "salary", "agg": "avg", "alias": "avg_salary"},
    {"column": "employee_id", "agg": "count_distinct", "alias": "headcount"}
  ],
  "group_by": ["department"],
  "filters": [],
  "time_range": null,
  "order_by": [{"column": "salary", "direction": "DESC"}],
  "limit": 25,
  "compare_to": null,
  "confidence": "high",
  "ambiguous_columns": [],
  "ranking_factors": [],
  "reasoning": "Comparison of two metrics across departments. Avg salary shows compensation level; count_distinct employee_id shows team size. Together they reveal compensation efficiency."
}

--- Example 4: Distribution ---
Question: What is the distribution of order values by category?
Schema: category (VARCHAR), order_total (DECIMAL)
Plan:
{
  "intent": "distribution",
  "metrics": [
    {"column": "order_total", "agg": "avg", "alias": "avg_order"},
    {"column": "order_total", "agg": "min", "alias": "min_order"},
    {"column": "order_total", "agg": "max", "alias": "max_order"},
    {"column": "order_total", "agg": "count", "alias": "order_count"}
  ],
  "group_by": ["category"],
  "filters": [],
  "time_range": null,
  "order_by": [{"column": "order_total", "direction": "DESC"}],
  "limit": 25,
  "compare_to": null,
  "confidence": "high",
  "ambiguous_columns": [],
  "ranking_factors": [],
  "reasoning": "Distribution analysis: showing avg, min, max and count of order values per category reveals spread and volume simultaneously."
}"""

# ─────────────────────────────────────────────────────────────────────────────
# SQL GENERATOR (free-form fallback)
# ─────────────────────────────────────────────────────────────────────────────
SQL_GENERATOR_SYSTEM = """You are a MySQL 8.0 expert SQL engineer generating precise, optimized analytical queries.

CRITICAL RULES (violating any will produce a broken query):
1. Output ONLY the SQL query — no explanation, no markdown, no backtick fences, no comments.
2. Use ONLY the table and column names provided in the schema. Never invent columns or tables.
3. Use backtick-quoted identifiers for all table and column names: `table_name`.`column_name`
4. Use MySQL 8.0 syntax: DATE_FORMAT, DATE_SUB, QUARTER, YEAR, MONTH, TIMESTAMPDIFF, window functions, CTEs.
5. MySQL has no MEDIAN, PERCENTILE_CONT, CORR, or FULL OUTER JOIN — use alternatives.
6. Strictly comply with ONLY_FULL_GROUP_BY: every SELECT column must be in GROUP BY or be an aggregate.
7. Always include a LIMIT clause (default 1000; honor explicit "top N" requests).
8. Never use subqueries returning multiple rows where a scalar is expected.
9. For growth/change calculations, use CTEs with LAG() window functions.
10. For percentage shares, use SUM(col) / SUM(SUM(col)) OVER () * 100.
11. For ranking, use RANK() or DENSE_RANK() window functions inside a CTE.
12. Filter NULLs explicitly when the question implies non-null values.
13. For date grouping, always use DATE_FORMAT for month/quarter bucketing.

Think about the query structure before writing:
- What tables are involved?
- What aggregation level (row-level, group-level, window)?
- Are CTEs needed for multi-step calculations?
- Does the question imply a WHERE filter on date, category, or status?"""

SQL_FEW_SHOT = """-- Question: Top 5 regions by revenue last quarter
SELECT `region`, SUM(`net_sales_amt`) AS `total_revenue`
FROM `orders`
WHERE `order_date` BETWEEN '2026-07-01' AND '2026-09-30'
GROUP BY `region`
ORDER BY `total_revenue` DESC
LIMIT 5;

-- Question: Monthly unique customers this year with growth rate
WITH monthly AS (
  SELECT DATE_FORMAT(`created_at`, '%Y-%m') AS `month`,
         COUNT(DISTINCT `customer_id`) AS `customers`
  FROM `customers`
  WHERE `created_at` >= '2026-01-01'
  GROUP BY DATE_FORMAT(`created_at`, '%Y-%m')
)
SELECT `month`, `customers`,
       LAG(`customers`) OVER (ORDER BY `month`) AS `prev_month`,
       ROUND((`customers` - LAG(`customers`) OVER (ORDER BY `month`))
             / NULLIF(LAG(`customers`) OVER (ORDER BY `month`), 0) * 100, 2) AS `growth_pct`
FROM monthly
ORDER BY `month`
LIMIT 24;

-- Question: Revenue share by category
SELECT `category`,
       SUM(`order_total`) AS `revenue`,
       ROUND(SUM(`order_total`) / SUM(SUM(`order_total`)) OVER () * 100, 2) AS `revenue_share_pct`
FROM `orders`
GROUP BY `category`
ORDER BY `revenue` DESC
LIMIT 20;

-- Question: Quarter-over-quarter revenue growth
WITH quarterly AS (
  SELECT YEAR(`order_date`) AS `yr`, QUARTER(`order_date`) AS `q`,
         SUM(`net_sales_amt`) AS `revenue`
  FROM `orders`
  GROUP BY YEAR(`order_date`), QUARTER(`order_date`)
)
SELECT `yr`, `q`, `revenue`,
       LAG(`revenue`) OVER (ORDER BY `yr`, `q`) AS `prev_revenue`,
       ROUND((`revenue` - LAG(`revenue`) OVER (ORDER BY `yr`, `q`))
             / NULLIF(LAG(`revenue`) OVER (ORDER BY `yr`, `q`), 0) * 100, 2) AS `growth_pct`
FROM quarterly
ORDER BY `yr`, `q`
LIMIT 20;"""

# ─────────────────────────────────────────────────────────────────────────────
# INSIGHT GENERATOR
# ─────────────────────────────────────────────────────────────────────────────
INSIGHT_SYSTEM = """You are a senior data analyst writing clear, actionable business insights for decision-makers.

Your task: Given query results and computed facts, deliver a rich, structured analytical narrative.

RULES — follow every one:
1. Use ONLY the exact numbers from the computed facts provided. Never invent or estimate numbers.
2. Write 3-6 sentences. Be specific — cite exact figures, percentages, and rankings.
3. Structure your response with:
   a. The headline finding (the single most important insight)
   b. Supporting context (2nd and 3rd most notable patterns)
   c. A business implication or recommended action (if clearly derivable)
4. Do NOT mention SQL, databases, queries, tables, or technical terms.
5. Use business-friendly language — write as if presenting to a non-technical executive.
6. Highlight anomalies, outliers, or surprising patterns when present.
7. For trend data: comment on direction (growing/declining/flat) and rate of change.
8. For comparisons: name the winner and quantify the gap.
9. For distributions: note concentration (is it top-heavy? evenly spread?).
10. Always provide a written answer even when a chart is also shown — the text should stand alone.
11. Format numbers clearly: use commas for thousands, 2 decimal places for currency/percentages.
12. If the data reveals a risk, flag it explicitly.

Output format: plain prose paragraphs — no bullet points, no headers, no markdown formatting."""

# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_schema_block(columns: list[dict[str, Any]], max_tokens: int = 2000) -> str:
    """
    Build a compact one-line-per-column schema string.
    Format: `col_name` (type) -- description [ex: val1, val2] [stats: min=x, max=y, null=z%]
    Truncates to fit within approximate token budget.
    """
    lines = []
    for col in columns:
        name = col.get("safe_name", col.get("original_name", ""))
        ctype = col.get("col_type", "")
        desc = col.get("description", "")
        examples = col.get("top_values", [])
        null_pct = col.get("null_pct")
        col_min = col.get("col_min")
        col_max = col.get("col_max")
        distinct = col.get("distinct_count")

        line = f"`{name}` ({ctype})"
        if desc:
            line += f" -- {desc[:70]}"
        if examples:
            ex_str = ", ".join(str(v) for v in examples[:4])
            line += f" [ex: {ex_str}]"
        # Add stats for numeric/date columns
        stats = []
        if col_min is not None and col_max is not None:
            stats.append(f"range: {col_min}–{col_max}")
        if distinct is not None and ctype in ("str", "bool"):
            stats.append(f"{distinct} distinct")
        if null_pct is not None and float(null_pct) > 0:
            stats.append(f"{null_pct:.0f}% null")
        if stats:
            line += f" [{', '.join(stats)}]"
        lines.append(line)

    # Approximate token count: 1 token ≈ 4 chars
    result = "\n".join(lines)
    if len(result) // 4 > max_tokens:
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
        # Include last 6 turns for richer conversational context
        history_lines = chat_history[-6:]
        user_parts.append("Recent conversation context:\n" + "\n".join(history_lines))

    user_parts.append(f"Available column schema:\n{schema_block}")

    if resolved_dates:
        user_parts.append(f"Resolved date context: {resolved_dates}")

    user_parts.append(f"User question: {question}")
    user_parts.append("Think carefully, then return the JSON plan only:")

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
        f"Analytical question: {question}",
    ]
    if error_context:
        user_parts.append(
            f"Previous attempt failed with error:\n{error_context}\n"
            "Carefully analyze the error, fix the root cause, and produce a corrected query:"
        )

    user_parts.append("Generate the MySQL 8.0 SELECT query:")
    return system, "\n\n".join(user_parts)


def build_insight_prompt(
    question: str,
    computed_facts: dict[str, Any],
    result_preview: str,
    result_shape: tuple[int, int] | None = None,
) -> tuple[str, str]:
    """Return (system_prompt, user_message) for insight generation."""
    import json
    facts_str = json.dumps(computed_facts, indent=2, default=str)

    shape_note = ""
    if result_shape:
        rows, cols = result_shape
        shape_note = f"\nResult dimensions: {rows} rows × {cols} columns"

    user = (
        f"Analytical question asked: {question}\n\n"
        f"Computed statistical facts (use ONLY these numbers):\n{facts_str}{shape_note}\n\n"
        f"Top result rows (for context):\n{result_preview}\n\n"
        "Write a clear, specific, business-focused analytical insight:"
    )
    return INSIGHT_SYSTEM, user
