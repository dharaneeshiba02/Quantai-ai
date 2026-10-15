/**
 * api/client.ts — Typed API client for QuantAI backend.
 * Handles REST calls and SSE streaming.
 */

// Use the Vite proxy in development and same-origin API routes in production.
// Set VITE_API_URL only when the API is hosted on a different origin.
const BASE_URL = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

// ─── Types ────────────────────────────────────────────────────────────────────

export interface Source {
  id: string;
  name: string;
  source_type: string;
  table_name: string;
  row_count: number;
  col_count: number;
  created_at: string;
}

export interface ColumnCatalogEntry {
  safe_name: string;
  original_name: string;
  col_type: string;
  null_pct: number | null;
  distinct_count: number | null;
  col_min: string | null;
  col_max: string | null;
  top_values: string[] | null;
  description: string | null;
  is_date_col: boolean;
  is_key_col: boolean;
  is_pii: boolean;
}

export interface SourceOverview {
  source_id: string;
  name: string;
  row_count: number;
  col_count: number;
  type_breakdown: Record<string, number>;
  null_high_columns: string[];
  date_columns: string[];
  suggested_questions: string[];
}

export interface DatasetSummary {
  source_id: string;
  name: string;
  row_count: number;
  col_count: number;
  scale: 'small' | 'medium' | 'large';
  suggested_pages: number;
  summary: string;
  highlights: string[];
  quality_notes: string[];
  metric_notes: string[];
  chart_summaries: { title: string; caption: string; kind: string }[];
}

export interface HistoryRun {
  id: string;
  question: string;
  status: string;
  confidence: string | null;
  verified: boolean;
  created_at: string;
  row_count: number | null;
}

export interface HistoryRunDetail extends HistoryRun {
  session_id: string | null;
  source_id: string | null;
  source_name: string | null;
  answer: string | null;
  sql: string | null;
  columns: string[];
  error: string | null;
}

export interface UploadResult {
  source_id: string;
  table_name: string;
  row_count: number;
  col_count: number;
  elapsed_sec: number;
  message: string;
}

export interface DocumentUploadResult {
  status: string;
  filename: string;
  chunk_count: number;
  message: string;
}

export interface HealthStatus {
  provider_reachable: boolean;
  model_available: boolean;
  embed_model_available: boolean;
  model: string;
  embed_model: string;
  error: string | null;
  fix: string | null;
  available_models: string[];
  database: { status: string };
  privacy_mode: string;
  schema_only_mode: boolean;
  pii_masking: boolean;
}

// ─── SSE Event Types ─────────────────────────────────────────────────────────

export type SSEEventType =
  | 'step_started'
  | 'step_completed'
  | 'plan'
  | 'sql'
  | 'result_preview'
  | 'chart'
  | 'verification'
  | 'clarification'
  | 'final'
  | 'error';

export interface SSEEvent {
  type: SSEEventType;
  run_id?: string;
  step?: string;
  // plan
  plan?: Record<string, any>;
  interpretation?: string;
  resolved_dates?: string;
  grounding_issues?: any[];
  // sql
  sql?: string;
  is_fallback?: boolean;
  // result
  rows?: Record<string, any>[];
  columns?: string[];
  total_rows?: number;
  // chart
  chart?: Record<string, any>;
  chart_type?: string;
  // verification
  verified?: boolean;
  details?: string;
  // clarification
  question?: string;
  // final
  session_id?: string;
  insight?: string;
  confidence?: string;
  warnings?: string[];
  trace?: any[];
  errors?: string[];
  is_fallback_sql?: boolean;
  columns_used?: string[];
  // error
  message?: string;
}

// ─── REST Helpers ─────────────────────────────────────────────────────────────

async function request<T>(
  method: string,
  path: string,
  body?: unknown,
): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json() as Promise<T>;
}

function uploadFile<T>(path: string, file: File, onProgress?: (pct: number) => void): Promise<T> {
  const form = new FormData();
  form.append('file', file);

  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `${BASE_URL}${path}`);
    if (onProgress) {
      xhr.upload.onprogress = (event) => {
        if (event.lengthComputable) onProgress(Math.round((event.loaded / event.total) * 100));
      };
    }
    xhr.onload = () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          resolve(JSON.parse(xhr.responseText) as T);
        } catch {
          reject(new Error('The server returned an invalid upload response'));
        }
      } else {
        let message = `HTTP ${xhr.status}`;
        try {
          message = JSON.parse(xhr.responseText).detail || message;
        } catch {
          if (xhr.statusText) message = xhr.statusText;
        }
        reject(new Error(message));
      }
    };
    xhr.onerror = () => reject(new Error('Upload failed. Check that the backend is running.'));
    xhr.send(form);
  });
}

async function downloadResponse(response: Response, fallbackFilename: string): Promise<void> {
  if (!response.ok) {
    const body = await response.text().catch(() => '');
    let detail = body || response.statusText;
    try {
      const payload = JSON.parse(body);
      detail = payload.detail || payload.message || detail;
    } catch { /* keep the plain-text response */ }
    throw new Error(detail || `Download failed (HTTP ${response.status})`);
  }

  const blob = await response.blob();
  if (!blob.size) throw new Error('The server returned an empty file. Please try again.');

  const disposition = response.headers.get('Content-Disposition') || '';
  const utf8Filename = disposition.match(/filename\*=UTF-8''([^;]+)/i)?.[1];
  const plainFilename = disposition.match(/filename="?([^";]+)"?/i)?.[1];
  let filename = fallbackFilename;
  try {
    if (utf8Filename) filename = decodeURIComponent(utf8Filename);
    else if (plainFilename) filename = plainFilename;
  } catch { /* use the safe fallback name */ }

  const objectUrl = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = objectUrl;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
}

// ─── Public API ───────────────────────────────────────────────────────────────

export const api = {
  health: () => request<HealthStatus>('GET', '/api/health'),

  sources: {
    list: () => request<Source[]>('GET', '/api/sources'),
    catalog: (id: string) => request<ColumnCatalogEntry[]>('GET', `/api/sources/${id}/catalog`),
    overview: (id: string) => request<SourceOverview>('GET', `/api/sources/${id}/overview`),
    summary: (id: string) => request<DatasetSummary>('GET', `/api/sources/${id}/summary`),
    report: async (id: string, title: string): Promise<void> => {
      const response = await fetch(`${BASE_URL}/api/sources/${encodeURIComponent(id)}/report`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title }),
      });
      await downloadResponse(response, `${title.trim().slice(0, 25)}.pdf`);
    },
    preview: (id: string, page = 0) =>
      request<{ rows: Record<string, any>[]; page: number; page_size: number }>(
        'GET',
        `/api/sources/${id}/preview?page=${page}&page_size=100`,
      ),
    delete: (id: string) => request<{ status: string }>('DELETE', `/api/uploads/${id}`),
  },

  upload: (file: File, onProgress?: (pct: number) => void) =>
    uploadFile<UploadResult>('/api/upload', file, onProgress),

  uploadDocument: (file: File, onProgress?: (pct: number) => void) =>
    uploadFile<DocumentUploadResult>('/api/docs/upload', file, onProgress),

  documentChat: async (question: string, signal?: AbortSignal): Promise<{ response: string }> => {
    const form = new FormData();
    form.append('query', question);
    const res = await fetch(`${BASE_URL}/api/docs/chat`, { method: 'POST', body: form, signal });
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(error.detail || `HTTP ${res.status}`);
    }
    return res.json() as Promise<{ response: string }>;
  },

  exportDocumentReport: async (content: string, format = 'pdf'): Promise<void> => {
    const form = new FormData();
    form.append('content', content);
    form.append('format_type', format);
    const response = await fetch(`${BASE_URL}/api/docs/report`, { method: 'POST', body: form });
    await downloadResponse(response, `Quantai_Analysis_Report.${format}`);
  },

  history: {
    list: (limit = 50) => request<HistoryRun[]>('GET', `/api/history?limit=${limit}`),
    get: (id: string) => request<HistoryRunDetail>('GET', `/api/history/${encodeURIComponent(id)}`),
    delete: (id: string) => request<{ status: string }>('DELETE', `/api/history/${id}`),
  },

  feedback: (run_id: string, feedback: 1 | -1, source_id?: string) =>
    request<{ status: string }>('POST', '/api/feedback', { run_id, feedback, source_id }),

  export: async (run_id: string, format: string): Promise<void> => {
    const response = await fetch(`${BASE_URL}/api/export/${encodeURIComponent(run_id)}?format=${encodeURIComponent(format)}`);
    await downloadResponse(response, `Quantai_Analysis.${format}`);
  },

  settings: () => request<Record<string, any>>('GET', '/api/settings'),

  /**
   * Stream a query via SSE. Returns a cleanup function.
   */
  query: (
    question: string,
    source_id: string | null,
    session_id: string | null,
    previous_run_id: string | null,
    onEvent: (event: SSEEvent) => void,
    onDone: () => void,
    onError: (err: Error) => void,
  ): (() => void) => {
    let cancelled = false;
    let controller: AbortController | null = new AbortController();

    (async () => {
      try {
        const res = await fetch(`${BASE_URL}/api/query`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, source_id, session_id, previous_run_id }),
          signal: controller?.signal,
        });

        if (!res.ok) throw new Error(`HTTP ${res.status}`);

        const reader = res.body!.getReader();
        const decoder = new TextDecoder();
        let buffer = '';

        while (!cancelled) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() ?? '';

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const evt: SSEEvent = JSON.parse(line.slice(6));
                onEvent(evt);
                if (evt.type === 'final' || evt.type === 'error') {
                  onDone();
                  return;
                }
              } catch {
                // malformed event, skip
              }
            }
          }
        }

        onDone();
      } catch (err: any) {
        if (err.name !== 'AbortError') {
          onError(err instanceof Error ? err : new Error(String(err)));
        }
      }
    })();

    return () => {
      cancelled = true;
      controller?.abort();
      controller = null;
    };
  },
};
