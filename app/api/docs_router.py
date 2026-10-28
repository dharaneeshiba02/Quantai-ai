from __future__ import annotations

import logging
from typing import Any, Optional
import uuid
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.rag.chatbot import add_document_to_store, get_chatbot_chain
from app.rag.ingestion import extract_text
from app.rag.report_gen import generate_report

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/docs", tags=["documents"])
UPLOAD_DIR = get_settings().uploads_dir
MAX_DOCUMENT_BYTES = 50 * 1024 * 1024
ALLOWED_DOCUMENT_TYPES = {".pdf", ".docx", ".txt"}


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = Path((file.filename or f"doc_{uuid.uuid4().hex}").replace("\\", "/")).name
    if Path(filename).suffix.lower() not in ALLOWED_DOCUMENT_TYPES:
        raise HTTPException(status_code=400, detail="Upload a PDF, DOCX, or TXT document")

    file_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{filename}"
    try:
        total_bytes = 0
        with file_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                total_bytes += len(chunk)
                if total_bytes > MAX_DOCUMENT_BYTES:
                    raise HTTPException(status_code=413, detail="Document too large (max 50 MB)")
                destination.write(chunk)

        document_text = await run_in_threadpool(extract_text, str(file_path))
        chunk_count = await run_in_threadpool(add_document_to_store, document_text, {"filename": filename})
        return {
            "status": "success",
            "filename": filename,
            "chunk_count": chunk_count,
            "message": f"Indexed {chunk_count} text sections from {filename}. You can now ask questions about it.",
        }
    except HTTPException:
        file_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        file_path.unlink(missing_ok=True)
        log.error("Document upload failed for %s (%s)", filename, type(exc).__name__)
        raise HTTPException(
            status_code=422,
            detail="Could not read or index this document. Check the file and local embedding model, then try again.",
        ) from exc


@router.get("/plot/{filename}")
async def get_plot(filename: str):
    plot_path = UPLOAD_DIR / filename
    if plot_path.exists():
        return FileResponse(plot_path)
    return JSONResponse(status_code=404, content={"detail": "Plot not found"})


@router.post("/chat")
async def chat(
    query: str = Form(...),
    session_id: Optional[str] = Form(None),
    source_id: Optional[str] = Form(None),
):
    if not query.strip():
        raise HTTPException(status_code=400, detail="Enter a question about your documents")
    chain = get_chatbot_chain()
    payload = {
        "question": query.strip(),
        "session_id": session_id or "default",
        "source_id": source_id,
    }
    response = await run_in_threadpool(chain.invoke, payload)
    return {"response": response["answer"], "session_id": payload["session_id"]}


@router.post("/report")
async def create_report(content: str = Form(...), format_type: str = Form(...)):
    format_type = format_type.strip().lower()
    media_types = {
        "pdf": "application/pdf",
        "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "csv": "text/csv; charset=utf-8",
    }
    if format_type not in media_types:
        raise HTTPException(status_code=400, detail="Choose PDF, DOCX, or CSV report format")

    report_filename = f"report_{uuid.uuid4().hex}.{format_type}"
    report_path = UPLOAD_DIR / report_filename
    await run_in_threadpool(generate_report, content, format_type, str(report_path))
    return FileResponse(
        path=report_path,
        filename=f"Quantai_Analysis_Report.{format_type}",
        media_type=media_types[format_type],
    )
