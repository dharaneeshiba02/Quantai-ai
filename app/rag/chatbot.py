"""
app/rag/chatbot.py — General-purpose document & dataset question-answering engine.

Features:
- Session-scoped conversation memory (no cross-user/cross-session contamination).
- Lazy vectorstore and embedding initialization (fails safely if Chroma/Ollama offline).
- Dataset-aware context injection (answers both tabular data and document queries).
- Factual grounding without fabricated answers.
- Preserves backward compatibility with legacy `chat_history`, `add_document_to_store`, and `get_chatbot_chain`.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from app.config import get_settings

log = logging.getLogger(__name__)

# Session-scoped in-memory histories: {session_id: [{"role": "User", "content": "..."}]}
_SESSION_HISTORIES: dict[str, list[dict[str, str]]] = {}

# Backward compatibility alias for legacy scripts
chat_history: list[dict[str, str]] = []

_vectorstore = None
_embeddings = None
_llm = None


def get_llm():
    """Lazy LLM instance initialization."""
    global _llm
    if _llm is None:
        settings = get_settings()
        from langchain_openai import ChatOpenAI
        _llm = ChatOpenAI(
            model=settings.groq_model,
            api_key=settings.groq_api_key,
            base_url=settings.groq_base_url,
            temperature=settings.llm_temperature_insight,
            timeout=settings.llm_timeout_sec,
        )
    return _llm


def get_embeddings():
    """Lazy embedding model initialization."""
    global _embeddings
    if _embeddings is None:
        settings = get_settings()
        from langchain_ollama import OllamaEmbeddings
        _embeddings = OllamaEmbeddings(
            model=settings.llm_embed_model,
            base_url=settings.ollama_base_url,
        )
    return _embeddings


def get_vectorstore():
    """Lazy Chroma vector store initialization."""
    global _vectorstore
    if _vectorstore is None:
        try:
            from langchain_chroma import Chroma
            _vectorstore = Chroma(
                collection_name="quantai_docs",
                embedding_function=get_embeddings(),
                persist_directory="data/chroma_db",
            )
        except Exception as exc:
            log.warning("Could not initialize Chroma vector store: %s", exc)
            return None
    return _vectorstore


# Backward-compatible lazy property
class _LazyLLMProxy:
    def invoke(self, *args, **kwargs):
        return get_llm().invoke(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(get_llm(), name)


llm = _LazyLLMProxy()


def get_session_history(session_id: str = "default") -> list[dict[str, str]]:
    """Retrieve chat history for a specific session."""
    return _SESSION_HISTORIES.setdefault(session_id, [])


def clear_session_history(session_id: str = "default") -> None:
    """Clear conversation history for a specific session."""
    _SESSION_HISTORIES.pop(session_id, None)
    if session_id == "default":
        chat_history.clear()


def add_document_to_store(text: str, metadata: dict | None = None) -> int:
    """Index a document into the Chroma vector store."""
    if not text.strip():
        raise ValueError("No readable text was found in this document")
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    store = get_vectorstore()
    if store is None:
        raise RuntimeError("Vector store is not available. Please verify embedding service.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.create_documents([text], metadatas=[metadata or {}])
    store.add_documents(chunks)
    return len(chunks)


def _load_dataset_context(source_id: str) -> str:
    """Retrieve structured catalog context for an active dataset."""
    try:
        from app.storage.database import readonly_session
        from app.storage.models import Source, ColumnCatalog
        from sqlalchemy import select

        with readonly_session() as session:
            src = session.get(Source, source_id)
            if not src:
                return ""
            cols = session.execute(
                select(ColumnCatalog).where(ColumnCatalog.source_id == source_id)
            ).scalars().all()

            col_lines = [
                f"- `{c.safe_name}` ({c.col_type}): {c.distinct_count} distinct values"
                for c in cols[:15]
            ]
            return (
                f"Active Dataset: {src.name} ({src.row_count:,} rows, {src.col_count} columns)\n"
                f"Columns:\n" + "\n".join(col_lines)
            )
    except Exception as exc:
        log.warning("Could not load dataset context for %s: %s", source_id, exc)
        return ""


class CustomChatbotChain:
    def invoke(self, inputs: dict[str, Any]) -> dict[str, Any]:
        question = str(inputs.get("question", "")).strip()
        session_id = str(inputs.get("session_id", "default"))
        source_id = inputs.get("source_id")

        if not question:
            return {"answer": "Please ask a question about your documents or datasets."}

        # 1. Retrieve document context from vectorstore if available
        doc_context = ""
        store = get_vectorstore()
        if store is not None:
            try:
                retriever = store.as_retriever(search_kwargs={"k": 3})
                docs = retriever.invoke(question)
                doc_context = "\n\n".join([d.page_content for d in docs])
            except Exception as exc:
                log.warning("Vector retrieval failed: %s", exc)

        # 2. Retrieve dataset context if source_id is provided
        dataset_context = _load_dataset_context(source_id) if source_id else ""

        # Combine context
        context_parts = []
        if dataset_context:
            context_parts.append(f"### Dataset Schema & Metadata:\n{dataset_context}")
        if doc_context:
            context_parts.append(f"### Relevant Document Sections:\n{doc_context}")

        combined_context = "\n\n".join(context_parts) if context_parts else "No specific document context found."

        # Session history
        history = get_session_history(session_id)
        history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-5:]])

        prompt = (
            "You are an expert, objective AI Data Analyst for QuantAI.\n"
            "Answer the question based STRICTLY on the provided context and dataset information.\n"
            "If the context does not contain the answer, say so honestly rather than inventing facts.\n\n"
            f"Context:\n{combined_context}\n\n"
            f"Chat History:\n{history_str}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        try:
            response = get_llm().invoke(prompt)
            answer = str(getattr(response, "content", response))
        except Exception as exc:
            log.warning("LLM call in CustomChatbotChain failed: %s", exc)
            answer = f"I could not complete the query with the model at this time ({exc})."

        # Update session history
        history.append({"role": "User", "content": question})
        history.append({"role": "Assistant", "content": answer})

        # Sync default history for backward compatibility
        if session_id == "default":
            chat_history.clear()
            chat_history.extend(history)

        return {"answer": answer}


def get_chatbot_chain():
    return CustomChatbotChain()
