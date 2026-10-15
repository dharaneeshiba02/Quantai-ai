/**
 * pages/ChatPage.tsx — Main chat interface with sidebar and message stream.
 */
import React, { useEffect, useRef, useState } from 'react';
import { MessageSquare, Database, Plus, Sparkles, BarChart3, Download, Loader2, AlertTriangle } from 'lucide-react';
import { Sidebar } from '../components/layout/Sidebar';
import { ChatInput } from '../components/chat/ChatInput';
import { MessageBubble } from '../components/chat/MessageBubble';
import { UploadDialog } from '../components/data/UploadDialog';
import { DatasetReportDialog } from '../components/data/DatasetReportDialog';
import { ThemeToggle } from '../components/ui/ThemeToggle';
import { DataPreviewPanel } from './panels/DataPreviewPanel';
import { HistoryPanel } from './panels/HistoryPanel';
import { SettingsPanel } from './panels/SettingsPanel';
import { useAppStore } from '../store/useAppStore';
import { useQuery } from '../hooks/useQuery';
import { api, type DatasetSummary } from '../api/client';
import { DEMO_OVERVIEW } from '../api/demo';

interface Props {
  onNavigate: (page: 'welcome' | 'chat') => void;
}

export const ChatPage: React.FC<Props> = ({ onNavigate }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const [datasetSummary, setDatasetSummary] = useState<DatasetSummary | null>(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [summaryError, setSummaryError] = useState<string | null>(null);
  const [reportDialogOpen, setReportDialogOpen] = useState(false);

  const {
    messages, clearMessages,
    selectedSourceId, setSelectedSourceId, setSourceOverview, sourceOverview,
    uploadDialogOpen, setUploadDialogOpen,
    activePanel, setActivePanel,
    activeStream,
    sources, demoMode, backendReachable, documents, documentMode, setDocumentMode,
    setMessages, setSessionId,
  } = useAppStore();

  const { runQuery, cancelQuery, isStreaming } = useQuery();

  const selectedSource = sources.find((s) => s.id === selectedSourceId);
  const suggestions = selectedSource
    ? ['What does this dataset tell us?', ...(sourceOverview?.suggested_questions || [])].slice(0, 4)
    : (sourceOverview?.suggested_questions || []);

  // Scroll to bottom on new messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Load overview when source changes
  useEffect(() => {
    if (documentMode) { setSourceOverview(null); return; }
    if (!selectedSourceId) { setSourceOverview(null); return; }
    if (demoMode || !backendReachable) {
      setSourceOverview(DEMO_OVERVIEW);
      return;
    }
    api.sources.overview(selectedSourceId)
      .then(setSourceOverview)
      .catch(() => setSourceOverview(DEMO_OVERVIEW));
  }, [selectedSourceId, demoMode, documentMode, backendReachable]);

  // Profile a dataset as soon as it is uploaded or selected so the workspace
  // opens with a useful summary instead of asking the user to start from zero.
  useEffect(() => {
    if (documentMode || !selectedSourceId || demoMode || !backendReachable) {
      setDatasetSummary(null);
      setSummaryLoading(false);
      setSummaryError(null);
      return;
    }

    let cancelled = false;
    setDatasetSummary(null);
    setSummaryError(null);
    setSummaryLoading(true);
    api.sources.summary(selectedSourceId)
      .then((summary) => { if (!cancelled) setDatasetSummary(summary); })
      .catch((error: unknown) => {
        if (!cancelled) setSummaryError(error instanceof Error ? error.message : 'Dataset summary could not be prepared.');
      })
      .finally(() => { if (!cancelled) setSummaryLoading(false); });

    return () => { cancelled = true; };
  }, [selectedSourceId, demoMode, documentMode, backendReachable]);

  const handleSuggestion = (q: string) => {
    setActivePanel('chat');
    runQuery(q);
  };

  const openHistoryRun = async (runId: string) => {
    const run = await api.history.get(runId);
    setDocumentMode(false);
    setSelectedSourceId(run.source_id);
    if (run.session_id) setSessionId(run.session_id);
    const timestamp = run.created_at ? new Date(run.created_at).getTime() : Date.now();
    setMessages([
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
    ]);
    setActivePanel('chat');
  };

  return (
    <div className="flex h-screen bg-[#080b12] overflow-hidden">
      {/* Sidebar */}
      <Sidebar onNavigate={onNavigate} />

      {/* Main content */}
      <div className="flex flex-col flex-1 overflow-hidden">
        {/* Top bar */}
        <header className="flex items-center justify-between px-5 lg:px-8 h-[72px] border-b border-white/[0.06] bg-[#090c13]/90 backdrop-blur-xl shrink-0">
          <div className="flex items-center gap-3">
            {documentMode ? (
              <>
                <div className="w-7 h-7 rounded-lg bg-amber-500/15 border border-amber-500/25 flex items-center justify-center">
                  <MessageSquare className="w-3.5 h-3.5 text-amber-300" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-white leading-none">Document chat</p>
                  <p className="text-[11px] text-slate-500 mt-0.5">{documents.length} document{documents.length === 1 ? '' : 's'} indexed</p>
                </div>
              </>
            ) : selectedSource ? (
              <>
                <div className="w-9 h-9 rounded-xl bg-indigo-500/10 border border-indigo-400/15 flex items-center justify-center">
                  <Database className="w-3.5 h-3.5 text-indigo-400" />
                </div>
                <div>
                  <p className="text-sm font-semibold text-slate-100 leading-none">{selectedSource.name}</p>
                  <p className="text-[11px] text-slate-500 mt-1">
                    {selectedSource.row_count?.toLocaleString()} rows · {selectedSource.col_count} columns
                  </p>
                </div>
              </>
            ) : (
              <p className="text-sm text-slate-500">No data source selected</p>
            )}
          </div>

          <div className="flex items-center gap-2">
            {documents.length > 0 && (
              <button
                onClick={() => { clearMessages(); setDocumentMode(!documentMode); }}
                className="btn-ghost text-xs rounded-lg border border-white/[0.06]"
              >
                {documentMode ? 'Use dataset chat' : 'Ask documents'}
              </button>
            )}
            {selectedSource && !documentMode && (
              <button
                onClick={() => setReportDialogOpen(true)}
                className="btn-ghost text-xs rounded-lg border border-indigo-300/15 text-indigo-200"
              >
                <Download className="h-3.5 w-3.5" /> Dataset report
              </button>
            )}
            {messages.length > 0 && (
              <button
                onClick={clearMessages}
                className="btn-ghost text-xs rounded-lg border border-white/[0.06]"
              >
                New Session
              </button>
            )}
            <button
              onClick={() => setUploadDialogOpen(true)}
              className="btn-primary text-sm py-2.5 rounded-xl"
            >
              <Plus className="w-4 h-4" />
              Upload Data
            </button>
            <ThemeToggle />
          </div>
        </header>

        {/* Panel content */}
        <div className="flex-1 overflow-hidden flex flex-col">
          {activePanel === 'chat' && (
            <>
              {/* Messages */}
              <div className="flex-1 overflow-y-auto px-4 sm:px-7 lg:px-10 py-8 space-y-8 bg-[radial-gradient(ellipse_at_50%_0%,rgba(99,102,241,0.055),transparent_58%)]">
                <div className="max-w-5xl w-full mx-auto space-y-8">
                {messages.length === 0 ? (
                  /* Empty state */
                  <div className="flex flex-col items-center justify-center min-h-[62vh] text-center gap-6 animate-fade-in py-10">
                    <div className="w-16 h-16 rounded-[22px] bg-gradient-to-br from-indigo-500/20 to-violet-500/10 border border-indigo-300/15 shadow-[0_16px_60px_rgba(79,70,229,0.16)] flex items-center justify-center">
                      <Sparkles className="w-7 h-7 text-indigo-300" />
                    </div>
                    <div className="max-w-xl">
                      <p className="text-[10px] font-semibold tracking-[0.22em] uppercase text-indigo-300/75 mb-3">
                        Private analytics workspace
                      </p>
                      <h2 className="text-3xl sm:text-4xl font-semibold tracking-tight text-white mb-3">
                        {documentMode
                          ? 'Ask your documents'
                          : selectedSource ? 'Explore your data' : 'Welcome to Quantai AI'}
                      </h2>
                      <p className="text-slate-400 leading-relaxed max-w-lg">
                        {documentMode
                          ? `Ask questions about ${documents.slice(-2).join(' and ')}.`
                          : selectedSource
                          ? `Ask a question about ${selectedSource.name} and get a clear, evidence-based answer with charts when they help.`
                          : 'Upload a CSV or Excel dataset to begin.'}
                      </p>
                    </div>

                    {selectedSource && !documentMode && (
                      <section className="w-full max-w-4xl rounded-3xl border border-white/[0.08] bg-[#0e1420]/90 p-5 text-left shadow-[0_20px_70px_rgba(0,0,0,.2)] sm:p-6">
                        <div className="flex flex-col gap-4 border-b border-white/[0.06] pb-4 sm:flex-row sm:items-center sm:justify-between">
                          <div className="flex items-center gap-3">
                            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-xl border border-indigo-300/15 bg-indigo-400/10">
                              <Sparkles className="h-4 w-4 text-indigo-300" />
                            </div>
                            <div>
                              <p className="text-[10px] font-semibold uppercase tracking-[0.17em] text-indigo-300">Automatic dataset summary</p>
                              <p className="mt-1 text-xs text-slate-500">Prepared locally from {selectedSource.name}</p>
                            </div>
                          </div>
                        </div>

                        {summaryLoading ? (
                          <div className="flex items-center gap-2 py-6 text-sm text-slate-400">
                            <Loader2 className="h-4 w-4 animate-spin text-indigo-300" />
                            Profiling the dataset and finding useful patterns…
                          </div>
                        ) : summaryError ? (
                          <div role="status" className="flex items-start gap-2 py-5 text-sm text-amber-200">
                            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-300" />
                            <span>{summaryError}</span>
                          </div>
                        ) : datasetSummary ? (
                          <>
                            <p className="max-w-3xl py-5 text-sm leading-7 text-slate-200">{datasetSummary.summary}</p>
                            <div className="mb-5 flex flex-wrap gap-2">
                              <span className="rounded-lg border border-white/[0.07] bg-white/[0.025] px-3 py-1.5 text-xs text-slate-300">
                                {datasetSummary.row_count.toLocaleString()} rows
                              </span>
                              <span className="rounded-lg border border-white/[0.07] bg-white/[0.025] px-3 py-1.5 text-xs text-slate-300">
                                {datasetSummary.col_count.toLocaleString()} columns
                              </span>
                              <span className="rounded-lg border border-indigo-300/15 bg-indigo-400/[0.06] px-3 py-1.5 text-xs capitalize text-indigo-200">
                                {datasetSummary.scale} dataset · {datasetSummary.suggested_pages} page{datasetSummary.suggested_pages === 1 ? '' : 's'} report
                              </span>
                            </div>
                            <div className="grid gap-3 md:grid-cols-2">
                              {datasetSummary.highlights.slice(0, 4).map((finding, index) => (
                                <div key={index} className="flex gap-2.5 rounded-xl border border-white/[0.05] bg-white/[0.02] p-3.5">
                                  <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-md bg-indigo-400/10 text-[10px] font-semibold text-indigo-300">{index + 1}</span>
                                  <p className="text-xs leading-5 text-slate-300">{finding}</p>
                                </div>
                              ))}
                            </div>
                            {datasetSummary.metric_notes.length > 0 && (
                              <div className="mt-4 rounded-xl border border-amber-300/10 bg-amber-300/[0.035] px-3.5 py-3 text-xs leading-5 text-amber-100/80">
                                <span className="font-semibold text-amber-200">Interpretation note: </span>
                                {datasetSummary.metric_notes.join(' ')}
                              </div>
                            )}
                            {datasetSummary.chart_summaries.length > 0 && (
                              <div className="mt-5 rounded-xl border border-cyan-300/10 bg-cyan-300/[0.035] p-3.5">
                                <div className="mb-2 flex items-center gap-2 text-xs font-medium text-cyan-200">
                                  <BarChart3 className="h-3.5 w-3.5" />
                                  Relevant plots will be included in the PDF
                                </div>
                                <p className="text-xs leading-5 text-slate-400">
                                  {datasetSummary.chart_summaries.map((chart) => chart.title).join(' · ')}
                                </p>
                              </div>
                            )}
                          </>
                        ) : null}
                      </section>
                    )}

                    {/* Suggestions grid */}
                    {suggestions.length > 0 && (
                      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 max-w-3xl w-full mt-1">
                        {suggestions.map((q, i) => (
                          <button
                            key={i}
                            onClick={() => handleSuggestion(q)}
                            className="group text-left p-4 rounded-2xl glass-hover text-sm text-slate-300 
                                       hover:text-white hover:-translate-y-0.5 transition-all animate-fade-in-up"
                            style={{ animationDelay: `${i * 60}ms` }}
                          >
                            {q}
                          </button>
                        ))}
                      </div>
                    )}

                    {!selectedSource && (
                      <button
                        onClick={() => setUploadDialogOpen(true)}
                        className="btn-primary px-8 py-3.5"
                      >
                        <Plus className="w-5 h-5" />
                        Upload your dataset
                      </button>
                    )}
                  </div>
                ) : (
                  messages.map((msg) => (
                    <MessageBubble
                      key={msg.id}
                      message={msg}
                      isStreaming={isStreaming && msg.id === messages[messages.length - 1]?.id && msg.role === 'assistant'}
                      activeStep={activeStream?.current_step}
                      completedSteps={activeStream?.completed_steps}
                    />
                  ))
                )}
                <div ref={messagesEndRef} />
                </div>
              </div>

              {/* Input */}
              <div className="px-4 sm:px-7 lg:px-10 pb-3 pt-3 shrink-0 border-t border-white/[0.045] bg-[#090c13]/75">
                <div className="max-w-5xl mx-auto">
                  <ChatInput
                    onSubmit={runQuery}
                    isLoading={isStreaming}
                    onCancel={cancelQuery}
                    suggestions={messages.length === 0 ? suggestions : []}
                    disabled={!selectedSourceId && !demoMode && !documentMode}
                  />
                </div>
              </div>
            </>
          )}

          {activePanel === 'data' && (
            <DataPreviewPanel sourceId={selectedSourceId} />
          )}

          {activePanel === 'history' && (
            <HistoryPanel
              onRerun={(q) => { setActivePanel('chat'); runQuery(q); }}
              onOpen={openHistoryRun}
            />
          )}

          {activePanel === 'settings' && (
            <SettingsPanel />
          )}
        </div>
      </div>

      {/* Upload dialog */}
      {uploadDialogOpen && (
        <UploadDialog onClose={() => setUploadDialogOpen(false)} />
      )}
      {reportDialogOpen && selectedSource && (
        <DatasetReportDialog
          sourceId={selectedSource.id}
          sourceName={selectedSource.name}
          onClose={() => setReportDialogOpen(false)}
        />
      )}
    </div>
  );
};
