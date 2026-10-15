# QuantAI — Local AI Data Analyst 🚀

> **Privacy-First, Local Multi-Agent Data Analysis Platform**  
> Talk to your data in plain English. No SQL required. 100% private, running locally on your hardware.

---

## 🌟 Key Features

- 🔒 **100% Private & Local**: Powered by local LLMs via Ollama (`qwen2.5-coder:7b` / `llama3.1:8b`). Zero data leaves your machine.
- 🤖 **LangGraph Multi-Agent Architecture**: Robust query planning, deterministic SQL compilation, and execution verification.
- 🛡️ **SQL Guard & Sandboxing**: AST-level SQL validation with `sqlglot` and dedicated read-only database execution.
- 📊 **Pandas Verification & Statistics**: Automated cross-checking of SQL aggregate metrics against source data with automatic Plotly chart generation.
- 📄 **Dataset Analysis**: CSV and Excel analysis with schema-aware planning, charts, and reports.
- 📑 **Automated Report Generation**: One-click professional exports to CSV, DOCX, and PDF.

---

## 🏗️ System Architecture

```text
User Query ──> React UI ──> FastAPI (/api/query)
                                │
                                ▼
                       LangGraph Supervisor
                                │
       ┌────────────────────────┼────────────────────────┐
       ▼                        ▼                        ▼
 Source Resolver        Embedding Selector        Query Planner
       │                        │                        │
       └────────────────────────┼────────────────────────┘
                                ▼
                      Deterministic SQL Compiler
                                │
                                ▼
                     SQL Guard (AST Validator)
                                │
                                ▼
                      Read-Only Execution (MySQL)
                                │
                                ▼
                     Pandas Cross-Check & Stats
                                │
                                ▼
                     Plotly Chart & Insights
                                │
                                ▼
                       SSE Streaming Response
```

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.ai/) with `qwen2.5-coder:7b` (or `llama3.1:8b`)
- MySQL 8.0+

### 2. Backend Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment variables
cp .env.example .env

# Initialize database schema
python scripts/load_sample_data.py

# Run FastAPI backend
uvicorn app.main:app --reload --port 8000
```

### 3. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

Visit `http://localhost:5173` to start chatting with your data!
