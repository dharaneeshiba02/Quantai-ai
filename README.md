# QuantAI AI

QuantAI AI is a local data-analysis application. Upload CSV or Excel files, connect a MySQL database, and ask questions in plain English. The application generates SQL, validates it, runs read-only analysis, and returns tables, charts, and written insights.

## What is local

- The primary language model is Ollama `qwen3.5`.
- The local recovery model is `qwen2.5-coder:7b`.
- Embeddings use Ollama `nomic-embed-text`.
- No Groq, OpenAI, or other hosted LLM API is required.
- Database access is local by default (`127.0.0.1`).

The application still needs Ollama and MySQL running on the same computer. Local does not mean those services start automatically.

## Features

- Plain-English questions over uploaded datasets and MySQL data
- LangGraph multi-agent query planning
- Read-only SQL execution with `sqlglot` validation
- Pandas verification for numeric results
- Plotly charts and dataset summaries
- CSV, Excel, PDF, DOCX, and spreadsheet exports
- Local document search with Chroma and Ollama embeddings
- Privacy controls for schema-only mode, PII masking, and sample-row limits

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or newer
- Node.js 18 or newer
- Ollama
- MySQL 8.0 or newer

Install the local Ollama models before starting the application:

```bash
ollama pull qwen3.5
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
```

Check that Ollama is running and the models are available:

```bash
ollama list
```

## Installation

From the project root:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

macOS or Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your local MySQL credentials. The important local-model settings are already configured:

```dotenv
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://127.0.0.1:11434
LLM_MODEL=qwen3.5
LLM_FALLBACK_MODEL=qwen2.5-coder:7b
LLM_EMBED_MODEL=nomic-embed-text
```

Do not commit `.env`. It contains local database credentials and is intentionally ignored by Git.

## Run the application

The easiest Windows option is:

```text
run_localhost.bat
```

Choose unified mode to serve the built frontend and API at `http://localhost:8000`. For development, run the backend and frontend separately.

Backend:

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Frontend, in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in development mode or `http://localhost:8000` in unified mode.

## MySQL setup

The application expects these local schemas and users by default:

- Database: `quantai_data` for imported/user data
- Database: `quantai_app` for application metadata
- Users: `quantai_readonly`, `quantai_import`, and `quantai_app`

Set the corresponding passwords in `.env`. If your MySQL setup is different, update the host, port, schema, user, and password values before starting the backend.

## Verify the installation

Check the local model configuration:

```bash
python -c "import asyncio; from app.llm.client import health_check; print(asyncio.run(health_check()))"
```

Run the automated tests:

```bash
python -m pytest tests/ -q
```

The health check should report `provider: ollama`, `model: qwen3.5`, `provider_reachable: True`, and `model_available: True`.

## Troubleshooting

### Ollama is unreachable

Start Ollama, then verify `http://127.0.0.1:11434` is available:

```bash
ollama serve
```

### The model is not found

Pull the exact model name used by `.env`:

```bash
ollama pull qwen3.5
```

### The frontend still shows an old version

Rebuild the production frontend and hard-refresh the browser:

```bash
cd frontend
npm run build
```

### MySQL connection errors

Confirm MySQL is running, then check the local connection values in `.env`. The backend cannot import or query data until the configured MySQL user has the required permissions.

## Project layout

```text
app/                 FastAPI backend and analysis pipeline
frontend/            React and TypeScript user interface
scripts/             Database and dataset utilities
tests/               Automated backend tests
data/                Local uploads, exports, and indexes
.env.example         Safe configuration template
run_localhost.bat    Windows launcher
```

## Current release

`v1.50` — local Qwen3.5 model integration and documentation cleanup.
