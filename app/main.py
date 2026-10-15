"""
app/main.py — FastAPI application factory with all routes, SSE streaming,
static file serving, and startup health checks.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import pandas as pd
import numpy as np
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

# Ensure telemetry is off before anything else
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = "disabled"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

from app.config import get_settings

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Request/Response models
# ─────────────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    source_id: Optional[str] = None
    session_id: Optional[str] = None
    previous_run_id: Optional[str] = None


class FeedbackRequest(BaseModel):
    run_id: str
    feedback: int  # 1=up, -1=down
    source_id: Optional[str] = None


class RerunRequest(BaseModel):
    run_id: str
    modified_plan: dict


class DatasetReportRequest(BaseModel):
    title: str = Field(min_length=1, max_length=25)


# ─────────────────────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    cfg = get_settings()

    app = FastAPI(
        title="QuantAI AI",
        description="MCP-powered local AI data analyst — your data never leaves this machine",
        version="1.2",
    )

    # CORS: localhost only
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            f"http://127.0.0.1:{cfg.app_port}",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Startup ──────────────────────────────────────────────────────────────
    @app.on_event("startup")
    async def startup():
        _setup_logging()
        log.info("QuantAI AI starting up...")
        await _init_database()
        health = await _check_ollama_health()
        if not health["provider_reachable"]:
            log.warning("⚠️  LLM provider not reachable — LLM features will not work")
            log.warning("   Fix: %s", health.get("fix", ""))
        elif not health["model_available"]:
            log.warning("⚠️  Model %s not found — %s", health["model"], health.get("fix", ""))
        else:
            log.info("✅ LLM provider ready: %s", health["model"])

    # ── Health ───────────────────────────────────────────────────────────────
    @app.get("/api/health")
    async def health_check():
        health = await _check_ollama_health()
        db_ok = await _check_db_health()
        return {
            **health,
            "database": db_ok,
            "privacy_mode": "provider-backed",
            "schema_only_mode": cfg.schema_only_mode,
            "pii_masking": cfg.pii_masking_enabled,
        }

    # ── Query (SSE streaming) ─────────────────────────────────────────────────
    @app.post("/api/query")
    async def run_query(request: QueryRequest):
        """Stream agent progress and final result as SSE events."""
        async def event_generator() -> AsyncIterator[dict]:
            run_id = str(uuid.uuid4())
            session_id = request.session_id or str(uuid.uuid4())

            try:
                yield {"data": json.dumps({"type": "step_started", "step": "supervisor", "run_id": run_id}, default=_json_default)}

                from app.graph.state import new_state, cache_get
                from app.graph.workflow import get_workflow

                # Load previous plan for follow-up
                previous_plan = None
                if request.previous_run_id:
                    previous_plan = await _load_previous_plan(request.previous_run_id)

                await _ensure_chat_session(session_id, request.source_id)

                # Build initial state
                chat_history = await _load_chat_history(session_id)
                state = new_state(
                    question=request.question,
                    session_id=session_id,
                    source_id=request.source_id,
                    chat_history=chat_history,
                    previous_plan=previous_plan,
                )
                state["run_id"] = run_id

                # Run workflow
                workflow = get_workflow()
                final_state = None
                last_step = None

                async for event in workflow.astream(state):
                    for node_name, node_state in event.items():
                        state.update(node_state)
                        step = node_state.get("current_step", node_name)
                        if step != last_step:
                            yield {"data": json.dumps({
                                "type": "step_completed",
                                "step": step,
                                "run_id": run_id,
                            }, default=_json_default)}
                            last_step = step

                        # Stream specific events
                        if node_state.get("query_plan"):
                            yield {"data": json.dumps({
                                "type": "plan",
                                "plan": node_state["query_plan"],
                                "interpretation": node_state.get("interpretation_text"),
                                "resolved_dates": node_state.get("resolved_date_description"),
                                "grounding_issues": node_state.get("grounding_issues", []),
                                "run_id": run_id,
                            }, default=_json_default)}

                        if node_state.get("generated_sql"):
                            yield {"data": json.dumps({
                                "type": "sql",
                                "sql": node_state["generated_sql"],
                                "is_fallback": node_state.get("is_fallback_sql", False),
                                "run_id": run_id,
                            }, default=_json_default)}

                        if node_state.get("result_row_count") is not None:
                            result_df = cache_get(run_id, "result_df")
                            preview = []
                            if result_df is not None:
                                preview = _df_to_preview(result_df, limit=50)
                            yield {"data": json.dumps({
                                "type": "result_preview",
                                "rows": preview,
                                "columns": node_state.get("result_columns", []),
                                "total_rows": node_state.get("result_row_count", 0),
                                "run_id": run_id,
                            }, default=_json_default)}

                        if node_state.get("chart_json"):
                            yield {"data": json.dumps({
                                "type": "chart",
                                "chart": node_state["chart_json"],
                                "chart_type": node_state.get("chart_type"),
                                "run_id": run_id,
                            }, default=_json_default)}

                        if node_state.get("cross_check_result"):
                            yield {"data": json.dumps({
                                "type": "verification",
                                "verified": node_state.get("verified", False),
                                "details": node_state["cross_check_result"].get("details"),
                                "run_id": run_id,
                            }, default=_json_default)}

                        if node_state.get("clarification_question"):
                            yield {"data": json.dumps({
                                "type": "clarification",
                                "question": node_state["clarification_question"],
                                "run_id": run_id,
                            }, default=_json_default)}

                        final_state = state

                # Final event
                if final_state:
                    result_df = cache_get(run_id, "result_df")
                    await _save_query_run(run_id, session_id, request.source_id, request.question, dict(final_state))
                    await _save_chat_turn(
                        session_id,
                        run_id,
                        request.question,
                        final_state.get("insight_text") or "",
                    )

                    yield {"data": json.dumps({
                        "type": "final",
                        "run_id": run_id,
                        "session_id": session_id,
                        "question": request.question,
                        "insight": final_state.get("insight_text"),
                        "sql": final_state.get("generated_sql"),
                        "interpretation": final_state.get("interpretation_text"),
                        "resolved_dates": final_state.get("resolved_date_description"),
                        "chart": final_state.get("chart_json"),
                        "chart_type": final_state.get("chart_type"),
                        "confidence": final_state.get("confidence"),
                        "verified": final_state.get("verified", False),
                        "warnings": final_state.get("warnings", []),
                        "trace": final_state.get("trace", []),
                        "grounding_issues": final_state.get("grounding_issues", []),
                        "columns_used": final_state.get("result_columns", []),
                        "is_fallback_sql": final_state.get("is_fallback_sql", False),
                        "errors": final_state.get("errors", []),
                    }, default=_json_default)}

            except Exception as exc:
                log.exception("Query stream error")
                yield {"data": json.dumps({
                    "type": "error",
                    "message": str(exc),
                    "run_id": run_id,
                }, default=_json_default)}

        return EventSourceResponse(event_generator())

    # ── Plan re-run (no LLM) ─────────────────────────────────────────────────
    @app.post("/api/query/rerun")
    async def rerun_plan(request: RerunRequest):
        """Re-run a modified plan without calling the LLM."""
        try:
            from app.accuracy.plan_to_sql import compile_plan, plan_to_interpretation
            from app.security.sql_guard import validate_sql
            from app.storage.database import readonly_session
            from sqlalchemy import text as sa_text

            # Load original run to get table info
            run_info = await _load_run(request.run_id)
            if not run_info:
                raise HTTPException(status_code=404, detail="Run not found")

            plan = request.modified_plan
            table_name = run_info.get("table_name", "")
            schema_name = run_info.get("schema_name", "quantai_data")

            sql = compile_plan(plan, table_name, schema_name)
            guard = validate_sql(sql)
            if not guard.valid:
                raise HTTPException(status_code=400, detail=guard.error)

            with readonly_session() as conn:
                result_df = pd.read_sql(sa_text(guard.sql), conn)

            interpretation = plan_to_interpretation(plan, table_name)
            preview = _df_to_preview(result_df, limit=100)

            return {
                "sql": guard.sql,
                "interpretation": interpretation,
                "rows": preview,
                "total_rows": len(result_df),
                "columns": list(result_df.columns),
            }
        except HTTPException:
            raise
        except Exception as exc:
            log.exception("Rerun error")
            raise HTTPException(status_code=500, detail=str(exc))

    # ── Upload ────────────────────────────────────────────────────────────────
    @app.post("/api/upload")
    async def upload_file(file: UploadFile = File(...)):
        """Upload CSV or Excel, import to MySQL, register as source."""
        cfg = get_settings()
        allowed = (".csv", ".xlsx", ".xls")
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in allowed:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {suffix}")

        filename = Path((file.filename or f"uploaded_{uuid.uuid4().hex}").replace("\\", "/")).name
        if not filename or filename in {".", ".."}:
            raise HTTPException(status_code=400, detail="Invalid file name")

        # Store each upload separately, and cap it while streaming to disk.
        upload_path = cfg.uploads_dir / f"{uuid.uuid4().hex}_{filename}"
        result = None

        try:
            total_bytes = 0
            with upload_path.open("wb") as destination:
                while chunk := await file.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > 200 * 1024 * 1024:
                        raise HTTPException(status_code=413, detail="File too large (max 200 MB)")
                    destination.write(chunk)
            log.info("Uploaded %s (%d bytes)", filename, total_bytes)

            from app.utils.import_pipeline import import_file
            result = await import_file(upload_path, filename)

            # Register source in DB
            source_id = await _register_source(
                name=filename,
                source_type="csv" if suffix == ".csv" else "excel",
                table_name=result["table_name"],
                schema_name=result["schema_name"],
                row_count=result["row_count"],
                col_count=result["col_count"],
                file_path=str(upload_path),
                catalog_entries=result["catalog_entries"],
            )

            return {
                "source_id": source_id,
                "table_name": result["table_name"],
                "row_count": result["row_count"],
                "col_count": result["col_count"],
                "elapsed_sec": result["elapsed_sec"],
                "message": f"Imported {result['row_count']:,} rows × {result['col_count']} columns in {result['elapsed_sec']:.1f}s",
            }
        except HTTPException:
            upload_path.unlink(missing_ok=True)
            raise
        except Exception as exc:
            upload_path.unlink(missing_ok=True)
            if result and result.get("table_name", "").startswith("upl_"):
                try:
                    from app.storage.database import import_connection
                    from sqlalchemy import text as sa_text
                    with import_connection() as conn:
                        conn.execute(sa_text(
                            f"DROP TABLE IF EXISTS `quantai_data`.`{result['table_name']}`"
                        ))
                        conn.commit()
                except Exception:
                    pass
            database_error = getattr(exc, "orig", exc)
            log.error(
                "Upload/import failed for %s (%s): %s",
                filename,
                type(exc).__name__,
                database_error,
            )
            raise HTTPException(
                status_code=422,
                detail=f"Could not import this spreadsheet: {database_error}",
            ) from exc

    # ── Sources ───────────────────────────────────────────────────────────────
    @app.get("/api/sources")
    async def list_sources():
        from app.storage.models import Source
        from app.storage.database import app_session
        from sqlalchemy import select
        with app_session() as session:
            sources = session.execute(
                select(Source).where(Source.is_active == True).order_by(Source.updated_at.desc())
            ).scalars().all()
            return [
                {
                    "id": s.id,
                    "name": s.name,
                    "source_type": s.source_type,
                    "table_name": s.table_name,
                    "row_count": s.row_count,
                    "col_count": s.col_count,
                    "created_at": s.created_at.isoformat(),
                }
                for s in sources
            ]

    @app.get("/api/sources/{source_id}/catalog")
    async def get_catalog(source_id: str):
        from app.storage.models import ColumnCatalog
        from app.storage.database import app_session
        from sqlalchemy import select
        with app_session() as session:
            cols = session.execute(
                select(ColumnCatalog)
                .where(ColumnCatalog.source_id == source_id)
                .order_by(ColumnCatalog.ordinal_position)
            ).scalars().all()
            return [
                {
                    "safe_name": c.safe_name,
                    "original_name": c.original_name,
                    "col_type": c.col_type,
                    "null_pct": c.null_pct,
                    "distinct_count": c.distinct_count,
                    "col_min": c.col_min,
                    "col_max": c.col_max,
                    "top_values": c.top_values,
                    "description": c.description,
                    "is_date_col": c.is_date_col,
                    "is_key_col": c.is_key_col,
                    "is_pii": c.is_pii,
                }
                for c in cols
            ]

    @app.get("/api/sources/{source_id}/overview")
    async def get_source_overview(source_id: str):
        from app.storage.models import Source, ColumnCatalog
        from app.storage.database import app_session
        from sqlalchemy import select
        with app_session() as session:
            src = session.get(Source, source_id)
            if not src:
                raise HTTPException(404, "Source not found")
            cols = session.execute(
                select(ColumnCatalog).where(ColumnCatalog.source_id == source_id)
            ).scalars().all()
            source_name = src.name
            source_row_count = src.row_count
            source_col_count = src.col_count
            catalog = [
                (col.col_type, col.null_pct, col.is_date_col, col.safe_name)
                for col in cols
            ]
            suggested_questions = _generate_suggestions(cols, src)

        type_counts = {}
        null_high = []
        date_cols = []
        for col_type, null_pct, is_date_col, safe_name in catalog:
            type_counts[col_type] = type_counts.get(col_type, 0) + 1
            if (null_pct or 0) > 20:
                null_high.append(safe_name)
            if is_date_col:
                date_cols.append(safe_name)

        return {
            "source_id": source_id,
            "name": source_name,
            "row_count": source_row_count,
            "col_count": source_col_count,
            "type_breakdown": type_counts,
            "null_high_columns": null_high,
            "date_columns": date_cols,
            "suggested_questions": suggested_questions,
        }

    @app.get("/api/sources/{source_id}/summary")
    async def get_dataset_summary(source_id: str):
        """Return an automatic profile so newly uploaded datasets have an immediate summary."""
        from app.reports.dataset_report import build_dataset_profile
        try:
            return await run_in_threadpool(build_dataset_profile, source_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="Dataset not found")
        except Exception as exc:
            log.exception("Dataset summary failed for source %s", source_id)
            raise HTTPException(status_code=500, detail="Could not summarize this dataset. Please try again.") from exc

    @app.post("/api/sources/{source_id}/report")
    async def download_dataset_report(source_id: str, request: DatasetReportRequest):
        """Generate a title-customized PDF report for a dataset."""
        title = request.title.strip()
        if not title:
            raise HTTPException(status_code=422, detail="Enter a report title")
        from app.reports.dataset_report import build_dataset_profile, write_dataset_report_pdf
        try:
            profile = await run_in_threadpool(build_dataset_profile, source_id)
            report_path = cfg.exports_dir / f"dataset_report_{uuid.uuid4().hex}.pdf"
            await run_in_threadpool(write_dataset_report_pdf, profile, title, report_path)
        except KeyError:
            raise HTTPException(status_code=404, detail="Dataset not found")
        except Exception as exc:
            log.exception("Dataset PDF report failed for source %s", source_id)
            raise HTTPException(status_code=500, detail="Could not create this PDF report. Please try again.") from exc

        from slugify import slugify
        filename = f"{slugify(title, separator='_') or 'dataset_report'}.pdf"
        return FileResponse(
            path=report_path,
            filename=filename,
            media_type="application/pdf",
            background=BackgroundTask(report_path.unlink, missing_ok=True),
        )

    @app.get("/api/sources/{source_id}/preview")
    async def get_source_preview(source_id: str, page: int = 0, page_size: int = 100):
        from app.storage.models import Source
        from app.storage.database import app_session, readonly_session
        from sqlalchemy import text as sa_text
        with app_session() as session:
            src = session.get(Source, source_id)
            if not src:
                raise HTTPException(404, "Source not found")
            table_name = src.table_name
        offset = page * page_size
        with readonly_session() as conn:
            df = pd.read_sql(
                sa_text(f"SELECT * FROM `quantai_data`.`{table_name}` LIMIT {page_size} OFFSET {offset}"),
                conn,
            )
        return {"rows": _df_to_preview(df), "page": page, "page_size": page_size}

    @app.delete("/api/uploads/{source_id}")
    async def delete_source(source_id: str):
        """Delete source: drops table, catalog, embeddings, examples, exports."""
        try:
            await _delete_source_fully(source_id)
        except Exception as exc:
            log.exception("Source deletion failed for %s", source_id)
            raise HTTPException(
                status_code=500,
                detail="This data source could not be deleted. It is still available; please try again.",
            ) from exc
        return {"status": "deleted"}

    # ── History ───────────────────────────────────────────────────────────────
    @app.get("/api/history")
    async def get_history(session_id: Optional[str] = None, limit: int = 50):
        from app.storage.models import QueryRun
        from app.storage.database import app_session
        from sqlalchemy import select
        with app_session() as session:
            q = select(QueryRun).order_by(QueryRun.created_at.desc()).limit(limit)
            if session_id:
                q = q.where(QueryRun.session_id == session_id)
            runs = session.execute(q).scalars().all()
            return [
                {
                    "id": r.id,
                    "question": r.question,
                    "status": r.status,
                    "confidence": r.confidence,
                    "verified": r.verified,
                    "created_at": r.created_at.isoformat(),
                    "row_count": r.row_count,
                }
                for r in runs
            ]

    @app.get("/api/history/{run_id}")
    async def get_history_run(run_id: str):
        """Return the saved answer and query metadata for opening a history item."""
        from app.storage.models import Message, QueryRun, Source
        from app.storage.database import app_session
        from sqlalchemy import select

        with app_session() as session:
            run = session.get(QueryRun, run_id)
            if not run:
                raise HTTPException(status_code=404, detail="History item not found")
            assistant_message = session.execute(
                select(Message)
                .where(Message.run_id == run_id, Message.role == "assistant")
                .order_by(Message.created_at.desc())
                .limit(1)
            ).scalars().first()
            source = session.get(Source, run.source_id) if run.source_id else None
            return {
                "id": run.id,
                "session_id": run.session_id,
                "source_id": run.source_id,
                "source_name": source.name if source else None,
                "question": run.question,
                "answer": assistant_message.content if assistant_message else run.insight_text,
                "sql": run.generated_sql,
                "status": run.status,
                "confidence": run.confidence,
                "verified": run.verified,
                "created_at": run.created_at.isoformat() if run.created_at else None,
                "row_count": run.row_count,
                "columns": run.columns_used or [],
                "error": run.error_message,
            }

    @app.delete("/api/history/{run_id}")
    async def delete_run(run_id: str):
        from app.storage.models import QueryRun
        from app.storage.database import app_session
        with app_session() as session:
            run = session.get(QueryRun, run_id)
            if run:
                session.delete(run)
        return {"status": "deleted"}

    # ── Feedback ─────────────────────────────────────────────────────────────
    @app.post("/api/feedback")
    async def submit_feedback(request: FeedbackRequest):
        from app.storage.models import QueryRun, ApprovedExample
        from app.storage.database import app_session
        with app_session() as session:
            run = session.get(QueryRun, request.run_id)
            if run:
                run.feedback = request.feedback
                if request.feedback == 1 and run.plan_json and request.source_id:
                    # Store as approved example
                    ex = ApprovedExample(
                        source_id=request.source_id,
                        question=run.question,
                        plan_json=run.plan_json,
                    )
                    session.add(ex)
        return {"status": "recorded"}

    # ── Settings ──────────────────────────────────────────────────────────────
    @app.get("/api/settings")
    async def get_settings_api():
        cfg = get_settings()
        return {
            "llm_model": cfg.llm_model,
            "llm_fallback_model": cfg.llm_fallback_model,
            "llm_num_ctx": cfg.llm_num_ctx,
            "schema_only_mode": cfg.schema_only_mode,
            "pii_masking_enabled": cfg.pii_masking_enabled,
            "accuracy_mode": cfg.accuracy_mode,
            "date_anchor": cfg.date_anchor,
            "column_selector_top_k": cfg.column_selector_top_k,
            "sample_rows_for_llm": cfg.sample_rows_for_llm,
        }

    # ── Export ────────────────────────────────────────────────────────────────
    @app.get("/api/export/{run_id}")
    async def export_result(run_id: str, format: str = "csv"):
        from app.export.router import handle_export
        return await handle_export(run_id, format)

    # ── Serve React frontend ──────────────────────────────────────────────────
    frontend_dist = cfg.frontend_dist
    if frontend_dist.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dist), html=True), name="frontend")
    else:
        @app.get("/")
        async def root():
            return {"message": "QuantAI AI API running. Build the frontend with: cd frontend && npm run build"}

    return app


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _check_ollama_health() -> dict:
    from app.llm.client import health_check
    return await health_check()


async def _check_db_health() -> dict:
    try:
        from app.storage.database import app_session
        from sqlalchemy import text
        with app_session() as session:
            session.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as exc:
        return {"status": "error", "message": str(exc)}


async def _init_database() -> None:
    """Create quantai_app tables if they don't exist."""
    try:
        from app.storage.database import Base, get_app_engine
        from app.storage import models  # noqa: ensure models are imported
        Base.metadata.create_all(get_app_engine())
        log.info("Database tables ready")
    except Exception as exc:
        log.error("Database init failed: %s", exc)


def _df_to_preview(df: "pd.DataFrame", limit: int = 200) -> list[dict]:
    """Convert DataFrame to JSON-serializable list of dicts."""
    return df.head(limit).where(pd.notna(df.head(limit)), None).to_dict("records")


def _json_default(value: Any) -> Any:
    """Serialize common Pandas, NumPy and SQL values used in streamed results."""
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Decimal):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, set):
        return list(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


async def _register_source(
    name: str,
    source_type: str,
    table_name: str,
    schema_name: str,
    row_count: int,
    col_count: int,
    file_path: Optional[str] = None,
    catalog_entries: Optional[list] = None,
) -> str:
    from app.storage.models import Source, ColumnCatalog
    from app.storage.database import app_session

    source_id = str(uuid.uuid4())
    with app_session() as session:
        src = Source(
            id=source_id,
            name=name,
            source_type=source_type,
            table_name=table_name,
            schema_name=schema_name,
            row_count=row_count,
            col_count=col_count,
            file_path=file_path,
        )
        session.add(src)

        if catalog_entries:
            for entry in catalog_entries:
                col = ColumnCatalog(
                    source_id=source_id,
                    **{k: v for k, v in entry.items() if k != "source_id"},
                )
                session.add(col)

    return source_id


async def _save_query_run(
    run_id: str,
    session_id: str,
    source_id: Optional[str],
    question: str,
    state: Any,
) -> None:
    from app.storage.models import QueryRun
    from app.storage.database import app_session
    try:
        with app_session() as db:
            run = QueryRun(
                id=run_id,
                session_id=session_id,
                source_id=source_id,
                question=question,
                plan_json=state.get("query_plan"),
                generated_sql=state.get("generated_sql"),
                status="success" if not state.get("errors") else "error",
                confidence=state.get("confidence"),
                verified=state.get("verified", False),
                row_count=state.get("result_row_count"),
                columns_used=state.get("result_columns", []),
                interpretation_text=state.get("interpretation_text"),
                insight_text=state.get("insight_text"),
                is_fallback_sql=state.get("is_fallback_sql", False),
                error_message="; ".join(state.get("errors", [])) or None,
            )
            db.add(run)
    except Exception as exc:
        log.warning("Failed to save query run (%s)", type(exc).__name__)


async def _ensure_chat_session(session_id: str, source_id: Optional[str]) -> None:
    """Create the local conversation row before persisting runs and messages."""
    from app.storage.models import ChatSession, Source
    from app.storage.database import app_session

    with app_session() as session:
        chat_session = session.get(ChatSession, session_id)
        if not chat_session:
            valid_source_id = source_id if source_id and session.get(Source, source_id) else None
            session.add(ChatSession(id=session_id, source_id=valid_source_id))
        elif source_id and chat_session.source_id is None and session.get(Source, source_id):
            chat_session.source_id = source_id


async def _save_chat_turn(session_id: str, run_id: str, question: str, answer: str) -> None:
    """Persist a user question and its final answer for history and follow-ups."""
    from app.storage.models import Message
    from app.storage.database import app_session

    try:
        with app_session() as session:
            session.add_all([
                Message(session_id=session_id, role="user", content=question, run_id=run_id),
                Message(session_id=session_id, role="assistant", content=answer, run_id=run_id),
            ])
    except Exception as exc:
        log.warning("Failed to save chat turn (%s)", type(exc).__name__)


async def _load_previous_plan(run_id: str) -> Optional[dict]:
    from app.storage.models import QueryRun
    from app.storage.database import app_session
    try:
        with app_session() as session:
            run = session.get(QueryRun, run_id)
            return run.plan_json if run else None
    except Exception:
        return None


async def _load_chat_history(session_id: str) -> list[str]:
    from app.storage.models import Message
    from app.storage.database import app_session
    from sqlalchemy import select
    try:
        with app_session() as session:
            msgs = session.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.created_at.desc())
                .limit(6)
            ).scalars().all()
            history = []
            for m in reversed(msgs):
                prefix = "User:" if m.role == "user" else "Assistant:"
                history.append(f"{prefix} {m.content[:200]}")
            return history
    except Exception:
        return []


async def _load_run(run_id: str) -> Optional[dict]:
    from app.storage.models import QueryRun, Source
    from app.storage.database import app_session
    try:
        with app_session() as session:
            run = session.get(QueryRun, run_id)
            if not run:
                return None
            src = session.get(Source, run.source_id) if run.source_id else None
            return {
                "table_name": src.table_name if src else "",
                "schema_name": src.schema_name if src else "quantai_data",
                "plan_json": run.plan_json,
            }
    except Exception:
        return None


async def _delete_source_fully(source_id: str) -> None:
    """Remove source, its table, catalog, embeddings, examples, exports."""
    from app.storage.models import ChatSession, QueryRun, Source
    from app.storage.database import app_session, import_connection
    from sqlalchemy import text
    from app.config import get_settings

    cfg = get_settings()

    with app_session() as session:
        src = session.get(Source, source_id)
        if not src:
            return
        table_name = src.table_name
        schema_name = src.schema_name or "quantai_data"
        file_path = src.file_path

    # Drop the uploaded table first so a privilege or connection failure leaves
    # the source metadata intact and lets the user retry safely.
    if table_name and table_name.startswith("upl_"):
        if not re.fullmatch(r"[A-Za-z0-9_]+", schema_name) or not re.fullmatch(r"upl_[A-Za-z0-9_]+", table_name):
            raise ValueError("Unsafe uploaded table identifier")
        with import_connection() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS `{schema_name}`.`{table_name}`"))
            conn.commit()

    # Query history remains available after its dataset is removed. Detach its
    # nullable source references before deleting the source row to satisfy the
    # database foreign keys.
    with app_session() as session:
        src = session.get(Source, source_id)
        if not src:
            return
        session.query(QueryRun).filter(QueryRun.source_id == source_id).update(
            {QueryRun.source_id: None}, synchronize_session=False
        )
        session.query(ChatSession).filter(ChatSession.source_id == source_id).update(
            {ChatSession.source_id: None}, synchronize_session=False
        )
        session.delete(src)

    from app.reports.dataset_report import invalidate_dataset_profile
    invalidate_dataset_profile(source_id)

    if file_path:
        uploaded_file = Path(file_path).resolve()
        uploads_root = cfg.uploads_dir.resolve()
        if uploaded_file.is_relative_to(uploads_root):
            try:
                uploaded_file.unlink(missing_ok=True)
            except OSError:
                log.warning("Could not remove uploaded file %s", uploaded_file, exc_info=True)

    # Remove embedding cache
    cache_file = cfg.embedding_cache_dir / f"{source_id}_cols.json"
    try:
        cache_file.unlink(missing_ok=True)
    except OSError:
        log.warning("Could not remove embedding cache %s", cache_file, exc_info=True)


def _generate_suggestions(cols: list, src: Any) -> list[str]:
    """Generate starter questions based on actual columns."""
    numeric_cols = [c.safe_name for c in cols if c.col_type in ("int", "float")]
    cat_cols = [c.safe_name for c in cols if c.col_type == "str"]
    date_cols = [c.safe_name for c in cols if c.is_date_col]

    suggestions = []
    if numeric_cols and date_cols:
        suggestions.append(f"What is the trend of {numeric_cols[0]} over time?")
    if numeric_cols and cat_cols:
        suggestions.append(f"Which {cat_cols[0]} has the highest {numeric_cols[0]}?")
    if numeric_cols:
        suggestions.append(f"What are the top 10 rows by {numeric_cols[0]}?")
    if date_cols:
        suggestions.append(f"How did {numeric_cols[0] if numeric_cols else 'count'} change last quarter?")

    suggestions.append("How many rows does this dataset have?")
    suggestions.append("Which columns have the most missing values?")
    return suggestions[:6]


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Suppress verbose library logs
    for logger_name in ["httpx", "httpcore", "sqlalchemy.engine", "langchain"]:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


# Entry point
app = create_app()


if __name__ == "__main__":
    import uvicorn
    cfg = get_settings()
    uvicorn.run("app.main:app", host=cfg.app_host, port=cfg.app_port, reload=True)
