/**
 * hooks/useQuery.ts — Custom hook for running queries against the backend or demo.
 */
import { useCallback } from 'react';
import { v4 as uuidv4 } from 'uuid';
import { api } from '../api/client';
import { streamDemoQuery } from '../api/demo';
import { useAppStore, type ChatMessage, type ActiveStream } from '../store/useAppStore';
import type { SSEEvent } from '../api/client';

export function useQuery() {
  const {
    sessionId, setSessionId,
    selectedSourceId,
    messages,
    addMessage, updateMessage,
    setActiveStream, activeStream,
    demoMode, documentMode,
  } = useAppStore();

  const runQuery = useCallback((question: string, previousRunId?: string) => {
    if (activeStream) {
      activeStream.cancel();
      setActiveStream(null);
    }

    const msgId = uuidv4();
    const currentSession = sessionId || uuidv4();
    const priorRunId = previousRunId || [...messages]
      .reverse()
      .find((message) => message.role === 'assistant' && message.run_id)?.run_id;
    if (!sessionId) setSessionId(currentSession);

    // Add user message
    addMessage({
      id: uuidv4(),
      role: 'user',
      content: question,
      status: 'done',
      timestamp: Date.now(),
    });

    // Add placeholder assistant message
    const assistantMsg: ChatMessage = {
      id: msgId,
      role: 'assistant',
      content: '',
      status: 'streaming',
      timestamp: Date.now(),
    };
    addMessage(assistantMsg);

    const handleEvent = (event: SSEEvent) => {
      const patch: Partial<ChatMessage> = {};

      if (event.type === 'step_completed' || event.type === 'step_started') {
        setActiveStream((prev: ActiveStream | null) => prev ? {
          ...prev,
          current_step: event.step || prev.current_step,
          completed_steps: event.type === 'step_completed'
            ? [...(prev.completed_steps || []), event.step || '']
            : prev.completed_steps,
        } : null);
        return;
      }

      if (event.type === 'sql') patch.sql = event.sql;
      if (event.type === 'result_preview') {
        patch.rows = event.rows;
        patch.columns = event.columns;
        patch.total_rows = event.total_rows;
      }
      if (event.type === 'chart') {
        patch.chart = event.chart;
        patch.chart_type = event.chart_type;
      }
      if (event.type === 'verification') {
        patch.verified = event.verified;
      }
      if (event.type === 'clarification') {
        patch.content = `**Clarification needed:** ${event.question}`;
        patch.status = 'done';
      }

      if (event.type === 'final') {
        patch.run_id = event.run_id;
        patch.session_id = event.session_id;
        patch.content = event.insight || '';
        patch.sql = patch.sql || event.sql;
        patch.chart = patch.chart || event.chart;
        patch.chart_type = patch.chart_type || event.chart_type;
        patch.confidence = event.confidence;
        patch.verified = event.verified;
        patch.warnings = event.warnings;
        patch.trace = event.trace;
        patch.is_fallback_sql = event.is_fallback_sql;
        patch.errors = event.errors;
        patch.status = (event.errors && event.errors.length > 0) ? 'error' : 'done';
        if (event.session_id) setSessionId(event.session_id);
      }

      if (event.type === 'error') {
        patch.content = `⚠️ Error: ${event.message}`;
        patch.status = 'error';
        patch.errors = [event.message || 'Unknown error'];
      }

      updateMessage(msgId, patch);
    };

    const handleDone = () => {
      setActiveStream(null);
      updateMessage(msgId, (prev: ChatMessage) => ({ status: prev.status === 'streaming' ? 'done' : prev.status }));
    };

    const handleError = (err: Error) => {
      updateMessage(msgId, {
        content: `⚠️ Connection error: ${err.message}`,
        status: 'error',
      });
      setActiveStream(null);
    };

    if (documentMode && !demoMode) {
      const controller = new AbortController();
      setActiveStream({
        run_id: msgId,
        current_step: 'searching_documents',
        completed_steps: [],
        cancel: () => controller.abort(),
      });
      api.documentChat(question, controller.signal)
        .then(({ response }) => {
          updateMessage(msgId, { content: response, status: 'done', is_document_answer: true });
        })
        .catch((err: unknown) => {
          if (err instanceof Error && err.name === 'AbortError') {
            updateMessage(msgId, { content: 'Document search stopped.', status: 'done' });
            return;
          }
          handleError(err instanceof Error ? err : new Error(String(err)));
        })
        .finally(() => setActiveStream(null));
      return;
    }

    let cancel: () => void;

    if (demoMode) {
      cancel = streamDemoQuery(question, handleEvent, handleDone);
    } else {
      cancel = api.query(
        question,
        selectedSourceId,
        currentSession,
        priorRunId || null,
        handleEvent,
        handleDone,
        handleError,
      );
    }

    setActiveStream({
      run_id: msgId,
      current_step: 'supervisor',
      completed_steps: [],
      cancel,
    });
  }, [activeStream, sessionId, selectedSourceId, messages, demoMode, documentMode, setActiveStream, setSessionId, addMessage, updateMessage]);

  const cancelQuery = useCallback(() => {
    if (activeStream) {
      activeStream.cancel();
      setActiveStream(null);
    }
  }, [activeStream, setActiveStream]);

  return { runQuery, cancelQuery, isStreaming: !!activeStream };
}
