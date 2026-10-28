/**
 * store/useAppStore.ts — Zustand global state store.
 * Manages sources, per-dataset / per-mode chat conversations, query state, and UI.
 */
import { create } from 'zustand';
import { v4 as uuidv4 } from 'uuid';
import { api, type Source, type SourceOverview } from '../api/client';

export type AppTheme = 'dark' | 'light';

const initialTheme: AppTheme = (() => {
  try {
    return window.localStorage.getItem('quantai.theme') === 'light' ? 'light' : 'dark';
  } catch {
    return 'dark';
  }
})();
document.documentElement.dataset.theme = initialTheme;
document.documentElement.style.colorScheme = initialTheme;

const savedDocuments = (() => {
  try {
    return JSON.parse(window.sessionStorage.getItem('quantai.documents') || '[]') as string[];
  } catch {
    return [];
  }
})();
const savedDocumentMode = window.sessionStorage.getItem('quantai.documentMode') === 'true';

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  run_id?: string;
  session_id?: string;
  sql?: string;
  chart?: Record<string, any>;
  chart_type?: string;
  rows?: Record<string, any>[];
  columns?: string[];
  total_rows?: number;
  confidence?: string;
  verified?: boolean;
  warnings?: string[];
  trace?: any[];
  is_fallback_sql?: boolean;
  is_document_answer?: boolean;
  errors?: string[];
  step?: string;
  status: 'streaming' | 'done' | 'error';
  timestamp: number;
}

export interface Conversation {
  sessionId: string;
  messages: ChatMessage[];
  sourceId: string | null;
  updatedAt: number;
}

export interface ActiveStream {
  run_id: string;
  current_step: string;
  completed_steps: string[];
  cancel: () => void;
}

interface AppStore {
  // Sources
  sources: Source[];
  selectedSourceId: string | null;
  sourceOverview: SourceOverview | null;
  sourcesLoaded: boolean;
  setSources: (sources: Source[]) => void;
  setSelectedSourceId: (id: string | null) => void;
  setSourceOverview: (ov: SourceOverview | null) => void;
  setSourcesLoaded: (v: boolean) => void;
  removeSource: (id: string) => void;
  addSource: (s: Source) => void;

  // Conversations & Chat
  sessionId: string | null;
  messages: ChatMessage[];
  conversations: Record<string, Conversation>;
  setSessionId: (id: string) => void;
  setMessages: (messages: ChatMessage[]) => void;
  addMessage: (msg: ChatMessage) => void;
  updateMessage: (id: string, patch: Partial<ChatMessage> | ((prev: ChatMessage) => Partial<ChatMessage>)) => void;
  clearMessages: () => void;
  switchChat: (targetSourceId: string | null, isDocMode?: boolean) => void;
  startNewChat: (sourceId?: string | null) => void;
  openHistoryRun: (runId: string) => Promise<void>;

  // Streaming
  activeStream: ActiveStream | null;
  setActiveStream: (s: ActiveStream | null | ((prev: ActiveStream | null) => ActiveStream | null)) => void;

  // UI
  sidebarOpen: boolean;
  activePanel: 'chat' | 'data' | 'history' | 'settings';
  uploadDialogOpen: boolean;
  setSidebarOpen: (v: boolean) => void;
  setActivePanel: (p: 'chat' | 'data' | 'history' | 'settings') => void;
  setUploadDialogOpen: (v: boolean) => void;

  // Local document RAG
  documents: string[];
  documentMode: boolean;
  addDocument: (name: string) => void;
  setDocumentMode: (v: boolean) => void;

  // Settings
  demoMode: boolean;
  backendReachable: boolean;
  theme: AppTheme;
  setDemoMode: (v: boolean) => void;
  setBackendReachable: (v: boolean) => void;
  setTheme: (theme: AppTheme) => void;
}

function getConversationKey(sourceId: string | null, isDocMode: boolean): string {
  if (isDocMode) return '__document_mode__';
  return sourceId ? `source_${sourceId}` : '__default__';
}

export const useAppStore = create<AppStore>((set, get) => ({
  // Sources
  sources: [],
  selectedSourceId: null,
  sourceOverview: null,
  sourcesLoaded: false,
  setSources: (sources) => set({ sources }),
  setSourceOverview: (ov) => set({ sourceOverview: ov }),
  setSourcesLoaded: (v) => set({ sourcesLoaded: v }),

  setSelectedSourceId: (id) => {
    const s = get();
    if (s.selectedSourceId !== id || s.documentMode) {
      s.switchChat(id, false);
    } else {
      set({ selectedSourceId: id });
    }
  },

  removeSource: (id) =>
    set((s) => {
      const sources = s.sources.filter((src) => src.id !== id);
      const removedSelectedSource = s.selectedSourceId === id;
      const nextSourceId = removedSelectedSource ? (sources[0]?.id ?? null) : s.selectedSourceId;
      const updatedConversations = { ...s.conversations };
      delete updatedConversations[`source_${id}`];

      const nextKey = getConversationKey(nextSourceId, false);
      const nextConv = updatedConversations[nextKey];

      return {
        sources,
        selectedSourceId: nextSourceId,
        sourceOverview: removedSelectedSource ? null : s.sourceOverview,
        messages: removedSelectedSource ? (nextConv?.messages || []) : s.messages,
        sessionId: removedSelectedSource ? (nextConv?.sessionId || null) : s.sessionId,
        conversations: updatedConversations,
      };
    }),

  addSource: (src) => set((s) => ({ sources: [src, ...s.sources] })),

  // Chat & Conversation Management
  sessionId: null,
  messages: [],
  conversations: {},

  setSessionId: (id) => set({ sessionId: id }),

  setMessages: (messages) =>
    set((s) => {
      const key = getConversationKey(s.selectedSourceId, s.documentMode);
      return {
        messages,
        conversations: {
          ...s.conversations,
          [key]: {
            sessionId: s.sessionId || uuidv4(),
            messages,
            sourceId: s.selectedSourceId,
            updatedAt: Date.now(),
          },
        },
      };
    }),

  addMessage: (msg) =>
    set((s) => {
      const messages = [...s.messages, msg];
      const key = getConversationKey(s.selectedSourceId, s.documentMode);
      const conv = s.conversations[key] || {
        sessionId: s.sessionId || uuidv4(),
        messages: [],
        sourceId: s.selectedSourceId,
        updatedAt: Date.now(),
      };
      return {
        messages,
        conversations: {
          ...s.conversations,
          [key]: {
            ...conv,
            sessionId: s.sessionId || conv.sessionId,
            messages,
            updatedAt: Date.now(),
          },
        },
      };
    }),

  updateMessage: (id, patch) =>
    set((s) => {
      const messages = s.messages.map((m) => {
        if (m.id !== id) return m;
        const resolved = typeof patch === 'function' ? patch(m) : patch;
        return { ...m, ...resolved };
      });
      const key = getConversationKey(s.selectedSourceId, s.documentMode);
      const conv = s.conversations[key] || {
        sessionId: s.sessionId || uuidv4(),
        messages: [],
        sourceId: s.selectedSourceId,
        updatedAt: Date.now(),
      };
      return {
        messages,
        conversations: {
          ...s.conversations,
          [key]: {
            ...conv,
            sessionId: s.sessionId || conv.sessionId,
            messages,
            updatedAt: Date.now(),
          },
        },
      };
    }),

  clearMessages: () => {
    const s = get();
    const key = getConversationKey(s.selectedSourceId, s.documentMode);
    const updatedConversations = { ...s.conversations };
    delete updatedConversations[key];
    set({
      messages: [],
      sessionId: uuidv4(),
      conversations: updatedConversations,
      activeStream: null,
    });
  },

  switchChat: (targetSourceId: string | null, isDocMode: boolean = false) => {
    const s = get();
    if (s.activeStream) {
      try {
        s.activeStream.cancel();
      } catch {
        /* ignore */
      }
    }

    // Save current conversation if it contains messages
    const currentKey = getConversationKey(s.selectedSourceId, s.documentMode);
    const updatedConversations = { ...s.conversations };
    if (s.messages.length > 0) {
      updatedConversations[currentKey] = {
        sessionId: s.sessionId || uuidv4(),
        messages: s.messages,
        sourceId: s.selectedSourceId,
        updatedAt: Date.now(),
      };
    }

    // Retrieve or initialize target conversation
    const targetKey = getConversationKey(targetSourceId, isDocMode);
    const existing = updatedConversations[targetKey];
    const targetSessionId = existing?.sessionId || uuidv4();
    const targetMessages = existing?.messages || [];

    set({
      conversations: updatedConversations,
      selectedSourceId: targetSourceId,
      documentMode: isDocMode,
      activePanel: 'chat',
      sessionId: targetSessionId,
      messages: targetMessages,
      activeStream: null,
    });
  },

  startNewChat: (sourceId?: string | null) => {
    const s = get();
    if (s.activeStream) {
      try {
        s.activeStream.cancel();
      } catch {
        /* ignore */
      }
    }

    const targetSourceId = sourceId !== undefined ? sourceId : s.selectedSourceId;
    const targetKey = getConversationKey(targetSourceId, s.documentMode);
    const newSessionId = uuidv4();

    const updatedConversations = {
      ...s.conversations,
      [targetKey]: {
        sessionId: newSessionId,
        messages: [],
        sourceId: targetSourceId,
        updatedAt: Date.now(),
      },
    };

    set({
      conversations: updatedConversations,
      selectedSourceId: targetSourceId,
      sessionId: newSessionId,
      messages: [],
      activePanel: 'chat',
      activeStream: null,
    });
  },

  openHistoryRun: async (runId: string) => {
    const s = get();
    if (s.activeStream) {
      try {
        s.activeStream.cancel();
      } catch {
        /* ignore */
      }
    }

    const run = await api.history.get(runId);
    const targetSourceId = run.source_id;
    const targetKey = getConversationKey(targetSourceId, false);
    const targetSessionId = run.session_id || uuidv4();
    const timestamp = run.created_at ? new Date(run.created_at).getTime() : Date.now();

    const loadedMessages: ChatMessage[] = [
      {
        id: `${run.id}-question`,
        role: 'user',
        content: run.question,
        status: 'done',
        timestamp,
      },
      {
        id: run.id,
        role: 'assistant',
        content: run.answer || run.error || 'The saved answer is unavailable. You can re-run this question.',
        run_id: run.id,
        session_id: run.session_id || undefined,
        sql: run.sql || undefined,
        columns: run.columns,
        total_rows: run.row_count ?? undefined,
        confidence: run.confidence || undefined,
        verified: run.verified,
        errors: run.status === 'error' && run.error ? [run.error] : undefined,
        status: run.status === 'error' ? 'error' : 'done',
        timestamp,
      },
    ];

    const updatedConversations = {
      ...s.conversations,
      [targetKey]: {
        sessionId: targetSessionId,
        messages: loadedMessages,
        sourceId: targetSourceId,
        updatedAt: Date.now(),
      },
    };

    set({
      conversations: updatedConversations,
      selectedSourceId: targetSourceId,
      documentMode: false,
      sessionId: targetSessionId,
      messages: loadedMessages,
      activePanel: 'chat',
      activeStream: null,
    });
  },

  // Streaming
  activeStream: null,
  setActiveStream: (s) =>
    set((state) => ({
      activeStream: typeof s === 'function' ? s(state.activeStream) : s,
    })),

  // UI
  sidebarOpen: true,
  activePanel: 'chat',
  uploadDialogOpen: false,
  setSidebarOpen: (v) => set({ sidebarOpen: v }),
  setActivePanel: (p) => set({ activePanel: p }),
  setUploadDialogOpen: (v) => set({ uploadDialogOpen: v }),

  documents: savedDocuments,
  documentMode: savedDocumentMode,
  addDocument: (name) =>
    set((s) => {
      const documents = s.documents.includes(name) ? s.documents : [...s.documents, name];
      window.sessionStorage.setItem('quantai.documents', JSON.stringify(documents));
      window.sessionStorage.setItem('quantai.documentMode', 'true');
      return { documents, documentMode: true, activePanel: 'chat' };
    }),

  setDocumentMode: (v) => {
    window.sessionStorage.setItem('quantai.documentMode', String(v));
    set({ documentMode: v });
  },

  // Settings
  demoMode: false,
  backendReachable: true,
  theme: initialTheme,
  setDemoMode: (v) => set({ demoMode: v, sourcesLoaded: false }),
  setBackendReachable: (v) => set({ backendReachable: v }),
  setTheme: (theme) => {
    try {
      window.localStorage.setItem('quantai.theme', theme);
    } catch {
      // The active theme still applies when persistent storage is unavailable.
    }
    document.documentElement.dataset.theme = theme;
    document.documentElement.style.colorScheme = theme;
    set({ theme });
  },
}));
