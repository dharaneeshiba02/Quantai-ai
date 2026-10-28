"""
scripts/evaluate.py — Evaluate system accuracy on domain-agnostic benchmark queries.
Dynamically discovers available sources and runs representative analytical questions.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from app.graph.state import new_state
from app.graph.workflow import get_workflow
from app.storage.database import app_session
from app.storage.models import Source, ColumnCatalog


def _build_test_queries_for_source(source: Source, cols: list[ColumnCatalog]) -> list[str]:
    """Dynamically craft representative queries based on the actual dataset schema."""
    queries = ["How many total rows are in this dataset?"]

    numeric_cols = [c.safe_name for c in cols if c.col_type in ("int", "float") and not c.is_key_col]
    cat_cols = [c.safe_name for c in cols if c.col_type == "str" and not c.is_key_col and (c.distinct_count or 0) <= 50]
    date_cols = [c.safe_name for c in cols if c.is_date_col]

    if numeric_cols and cat_cols:
        queries.append(f"What is the average {numeric_cols[0]} by {cat_cols[0]}?")
        queries.append(f"Which {cat_cols[0]} has the highest total {numeric_cols[0]}?")

    if numeric_cols and date_cols:
        queries.append(f"Show the trend of {numeric_cols[0]} over time.")

    if numeric_cols:
        queries.append(f"What are the top 5 records by {numeric_cols[0]}?")

    return queries[:5]


async def evaluate(source_id: str | None = None):
    print("=" * 60)
    print("Starting Domain-Agnostic Accuracy Evaluation")
    print("=" * 60)

    workflow = get_workflow()

    with app_session() as session:
        if source_id:
            src = session.query(Source).filter(Source.id == source_id).first()
            sources = [src] if src else []
        else:
            sources = session.query(Source).filter(Source.is_active.is_(True)).all()

        if not sources:
            print("No active source found. Import a dataset or run load_sample_data.py first.")
            return

        print(f"Found {len(sources)} source(s) to evaluate.")

        all_results = []
        for src in sources:
            cols = session.query(ColumnCatalog).filter(ColumnCatalog.source_id == src.id).all()
            test_queries = _build_test_queries_for_source(src, cols)

            print(f"\n--- Evaluating Source: {src.name} (table: {src.table_name}) ---")
            for q in test_queries:
                print(f"\nQuery: {q}")
                session_id = str(uuid.uuid4())
                state = new_state(
                    question=q,
                    session_id=session_id,
                    source_id=src.id,
                )

                try:
                    final_state = None
                    async for event in workflow.astream(state):
                        for node_name, node_state in event.items():
                            state.update(node_state)
                            final_state = state

                    if final_state is None:
                        final_state = {}

                    sql = final_state.get("generated_sql")
                    verified = final_state.get("verified", False)
                    rows = final_state.get("result_row_count", 0)
                    print(f"  SQL Generated: {sql is not None}")
                    print(f"  Rows Returned: {rows}")
                    print(f"  Verified     : {verified}")

                    all_results.append({
                        "source": src.name,
                        "question": q,
                        "sql_generated": sql is not None,
                        "verified": verified,
                        "rows": rows,
                    })
                except Exception as exc:
                    print(f"  Error: {exc}")
                    all_results.append({
                        "source": src.name,
                        "question": q,
                        "sql_generated": False,
                        "verified": False,
                        "error": str(exc),
                    })

    print("\n" + "=" * 60)
    print("Benchmark Results Summary")
    print("=" * 60)
    total = len(all_results)
    if total > 0:
        success = sum(1 for r in all_results if r.get("sql_generated"))
        verified = sum(1 for r in all_results if r.get("verified"))
        print(f"Total Queries Evaluated: {total}")
        print(f"SQL Generation Rate    : {success}/{total} ({success/total*100:.1f}%)")
        print(f"Cross-Check Verified   : {verified}/{total} ({verified/total*100:.1f}%)")
    else:
        print("No queries evaluated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate system on domain-agnostic queries.")
    parser.add_argument("--source-id", default=None, help="Optional source ID to evaluate")
    args = parser.parse_args()

    asyncio.run(evaluate(args.source_id))
