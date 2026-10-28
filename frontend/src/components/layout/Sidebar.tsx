/**
 * components/layout/Sidebar.tsx — Left sidebar with multi-chat switching,
 * new chat actions, data sources, and recent conversations.
 */
import React, { useEffect, useState } from 'react';
import {
  Database, MessageSquare, History, Settings, Upload,
  ChevronLeft, ChevronRight, Trash2, Plus,
  Wifi, WifiOff, Star, LayoutDashboard, Loader2
} from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { api, type HistoryRun } from '../../api/client';
import { DEMO_SOURCES } from '../../api/demo';

interface Props {
  onNavigate: (page: 'welcome' | 'chat') => void;
}

export const Sidebar: React.FC<Props> = ({ onNavigate }) => {
  const [deletingSourceId, setDeletingSourceId] = useState<string | null>(null);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [recentRuns, setRecentRuns] = useState<HistoryRun[]>([]);

  const {
    sidebarOpen, setSidebarOpen,
    activePanel, setActivePanel,
    sources, setSources, sourcesLoaded, setSourcesLoaded,
    selectedSourceId,
    setUploadDialogOpen,
    demoMode, backendReachable, setBackendReachable,
    documentMode,
    switchChat, startNewChat, openHistoryRun,
    messages,
  } = useAppStore();

  // Load sources on mount
  useEffect(() => {
    if (demoMode) {
      if (!sourcesLoaded) {
        setSources(DEMO_SOURCES);
        switchChat(DEMO_SOURCES[0]?.id ?? null, false);
        setSourcesLoaded(true);
      }
      return;
    }
    if (sourcesLoaded || !backendReachable) return;
    api.sources.list()
      .then((loadedSources) => {
        setSources(loadedSources);
        setBackendReachable(true);
        if (loadedSources.length > 0 && !selectedSourceId) {
          switchChat(loadedSources[0].id, false);
        }
      })
      .catch(() => {
        setSources([]);
        setBackendReachable(false);
      })
      .finally(() => setSourcesLoaded(true));
  }, [demoMode, backendReachable, sourcesLoaded, setSources, setSourcesLoaded, setBackendReachable]);

  // Refresh recent queries whenever chat changes
  useEffect(() => {
    if (!backendReachable || demoMode) return;
    let active = true;
    api.history.list(6)
      .then((items) => {
        if (active) setRecentRuns(items);
      })
      .catch(() => {
        if (active) setRecentRuns([]);
      });
    return () => { active = false; };
  }, [backendReachable, demoMode, messages.length]);

  const handleDeleteSource = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    e.preventDefault();
    if (!window.confirm('Delete this data source? This cannot be undone.')) return;
    setDeleteError(null);
    setDeletingSourceId(id);
    try {
      const state = useAppStore.getState();
      if (state.selectedSourceId === id) {
        state.activeStream?.cancel();
        state.setActiveStream(null);
      }
      if (!demoMode) await api.sources.delete(id);
      useAppStore.getState().removeSource(id);
    } catch (err: unknown) {
      setDeleteError(err instanceof Error ? err.message : 'Could not delete this data source. Please try again.');
    } finally {
      setDeletingSourceId(null);
    }
  };

  const navItems = [
    { id: 'chat' as const, icon: MessageSquare, label: 'Chat' },
    { id: 'data' as const, icon: LayoutDashboard, label: 'Data' },
    { id: 'history' as const, icon: History, label: 'History' },
    { id: 'settings' as const, icon: Settings, label: 'Settings' },
  ];

  return (
    <aside
      className={`relative flex flex-col shrink-0 border-r border-slate-800/60 bg-[#080817] transition-all duration-300 ease-in-out ${
        sidebarOpen ? 'w-64' : 'w-16'
      }`}
    >
      {/* Top Header / Logo */}
      <div className="flex items-center gap-3 px-4 h-16 border-b border-slate-800/60 shrink-0">
        <div
          className="w-8 h-8 rounded-xl bg-indigo-600 flex items-center justify-center shrink-0 cursor-pointer shadow-lg shadow-indigo-600/30"
          onClick={() => onNavigate('welcome')}
          title="Home"
        >
          <Database className="w-4.5 h-4.5 text-white w-[18px] h-[18px]" />
        </div>
        {sidebarOpen && (
          <span className="font-bold text-white text-lg tracking-tight truncate">Quantai AI</span>
        )}

        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          className="ml-auto p-1 rounded-lg text-slate-600 hover:text-slate-300 hover:bg-slate-800 transition-colors shrink-0"
          title={sidebarOpen ? 'Collapse sidebar' : 'Expand sidebar'}
        >
          {sidebarOpen ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
        </button>
      </div>

      {/* New Chat Button */}
      <div className="px-3 pt-3 pb-1 shrink-0">
        {sidebarOpen ? (
          <button
            onClick={() => {
              startNewChat();
              onNavigate('chat');
            }}
            className="w-full flex items-center justify-center gap-2 py-2 px-3 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-medium text-xs shadow-md shadow-indigo-600/25 transition-all active:scale-[0.98]"
            title="Start a fresh chat conversation"
          >
            <Plus className="w-4 h-4" />
            <span>New Chat</span>
          </button>
        ) : (
          <button
            onClick={() => {
              startNewChat();
              onNavigate('chat');
            }}
            className="w-10 h-10 mx-auto flex items-center justify-center rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white shadow-md shadow-indigo-600/25 transition-all active:scale-[0.98]"
            title="Start a fresh chat conversation"
          >
            <Plus className="w-4 h-4" />
          </button>
        )}
      </div>

      {/* Demo / Status badge */}
      {sidebarOpen && (demoMode || !backendReachable) && (
        <div className="mx-3 mt-2 px-3 py-1.5 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center gap-2 shrink-0">
          <Star className="w-3.5 h-3.5 text-amber-400 shrink-0" />
          <span className="text-xs text-amber-300 font-medium">Demo Mode</span>
        </div>
      )}

      {/* Nav */}
      <nav className="flex flex-col gap-1 p-2 mt-1 shrink-0">
        {navItems.map(({ id, icon: Icon, label }) => (
          <button
            key={id}
            onClick={() => {
              setActivePanel(id);
              onNavigate('chat');
            }}
            className={`flex items-center gap-3 px-3 py-2 rounded-xl transition-all text-sm font-medium
              ${activePanel === id
                ? 'bg-indigo-500/15 text-indigo-300 border border-indigo-500/25'
                : 'text-slate-500 hover:text-slate-200 hover:bg-slate-800/60'
              }`}
            title={!sidebarOpen ? label : undefined}
          >
            <Icon className="w-4.5 h-4.5 w-[18px] h-[18px] shrink-0" />
            {sidebarOpen && <span>{label}</span>}
          </button>
        ))}
      </nav>

      {/* Divider */}
      <div className="h-px bg-slate-800/60 mx-3 my-1 shrink-0" />

      {deleteError && sidebarOpen && (
        <div role="alert" className="mx-3 mb-2 rounded-xl border border-red-400/15 bg-red-400/[0.06] px-3 py-2 text-[11px] leading-5 text-red-200 shrink-0">
          {deleteError}
        </div>
      )}

      {/* Sources list (Expanded) */}
      {sidebarOpen && (
        <div className="flex flex-col flex-1 overflow-hidden px-3">
          <div className="flex items-center justify-between mb-1.5 shrink-0">
            <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider">
              Data Sources (Chats)
            </span>
            <button
              onClick={() => setUploadDialogOpen(true)}
              className="p-1 rounded-lg text-slate-500 hover:text-indigo-400 hover:bg-slate-800 transition-colors"
              title="Upload new dataset"
            >
              <Plus className="w-3.5 h-3.5" />
            </button>
          </div>

          <div className="flex-1 overflow-y-auto space-y-1 min-h-0 pr-0.5">
            {sources.length === 0 ? (
              <button
                onClick={() => setUploadDialogOpen(true)}
                className="w-full flex flex-col items-center gap-2 p-4 rounded-xl border border-dashed border-slate-800 
                           text-slate-600 hover:text-slate-400 hover:border-slate-700 transition-colors"
              >
                <Upload className="w-5 h-5" />
                <span className="text-xs">Upload CSV or Excel</span>
              </button>
            ) : (
              sources.map((src) => {
                const isActive = selectedSourceId === src.id && !documentMode;
                return (
                  <div
                    key={src.id}
                    onClick={() => {
                      switchChat(src.id, false);
                      onNavigate('chat');
                    }}
                    className={`group flex items-center gap-2.5 px-3 py-2 rounded-xl cursor-pointer transition-all border
                      ${isActive
                        ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-100 shadow-sm shadow-indigo-500/10'
                        : 'border-transparent hover:bg-slate-800/60 text-slate-400 hover:text-slate-200'
                      }`}
                  >
                    <div className={`w-6 h-6 rounded-md flex items-center justify-center shrink-0 text-[10px] font-bold
                      ${src.source_type === 'csv' ? 'bg-emerald-500/15 text-emerald-400' : 'bg-blue-500/15 text-blue-400'}
                    `}>
                      {src.source_type === 'csv' ? 'CSV' : 'XLS'}
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className={`text-xs font-medium truncate ${isActive ? 'text-indigo-100 font-semibold' : ''}`}>
                        {src.name.replace(/\.(csv|xlsx|xls)$/i, '')}
                      </p>
                      <p className="text-[10px] text-slate-500">{src.row_count?.toLocaleString()} rows</p>
                    </div>
                    <button
                      type="button"
                      onClick={(e) => handleDeleteSource(src.id, e)}
                      disabled={deletingSourceId === src.id}
                      aria-label={`Delete ${src.name}`}
                      title={deletingSourceId === src.id ? 'Deleting data source...' : 'Delete data source'}
                      className={`shrink-0 p-1.5 rounded-lg transition-all ${
                        isActive
                          ? 'opacity-70 hover:opacity-100 text-slate-400 hover:text-red-400 hover:bg-red-500/15'
                          : 'opacity-0 group-hover:opacity-70 hover:!opacity-100 text-slate-500 hover:text-red-400 hover:bg-slate-700/50'
                      } disabled:opacity-100 disabled:cursor-wait`}
                    >
                      {deletingSourceId === src.id
                        ? <Loader2 className="w-3.5 h-3.5 animate-spin text-indigo-400" />
                        : <Trash2 className="w-3.5 h-3.5" />}
                    </button>
                  </div>
                );
              })
            )}

            {/* Recent Questions section */}
            {recentRuns.length > 0 && (
              <div className="mt-3 pt-3 border-t border-slate-800/60">
                <span className="text-[10px] font-semibold text-slate-500 uppercase tracking-wider mb-1.5 block px-1">
                  Recent Questions
                </span>
                <div className="space-y-0.5">
                  {recentRuns.slice(0, 5).map((run) => (
                    <button
                      key={run.id}
                      onClick={() => {
                        openHistoryRun(run.id);
                        onNavigate('chat');
                      }}
                      className="w-full text-left px-2 py-1.5 rounded-lg text-[11px] text-slate-400 hover:text-slate-100 hover:bg-slate-800/60 truncate flex items-center gap-2 transition-colors"
                      title={run.question}
                    >
                      <MessageSquare className="w-3 h-3 text-indigo-400/70 shrink-0" />
                      <span className="truncate">{run.question}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Collapsed: icon only sources indicator */}
      {!sidebarOpen && (
        <div className="flex-1 flex flex-col items-center gap-1.5 px-2 mt-1 overflow-y-auto">
          <button
            onClick={() => setUploadDialogOpen(true)}
            className="w-full p-2 flex justify-center rounded-xl text-slate-500 hover:text-indigo-400 hover:bg-slate-800 transition-colors"
            title="Upload data"
          >
            <Upload className="w-4 h-4" />
          </button>
          {sources.map((src) => {
            const isActive = selectedSourceId === src.id && !documentMode;
            return (
              <button
                key={src.id}
                onClick={() => {
                  switchChat(src.id, false);
                  onNavigate('chat');
                }}
                title={src.name}
                className={`w-9 h-9 rounded-xl flex items-center justify-center text-[10px] font-bold transition-all border
                  ${isActive
                    ? 'bg-indigo-500/25 text-indigo-200 border-indigo-500/50 shadow-md shadow-indigo-500/20'
                    : 'border-transparent text-slate-400 hover:bg-slate-800'
                  }
                  ${src.source_type === 'csv' ? 'text-emerald-400' : 'text-blue-400'}`}
              >
                {src.source_type === 'csv' ? 'C' : 'X'}
              </button>
            );
          })}
        </div>
      )}

      {/* Status footer */}
      <div className={`shrink-0 px-3 py-3 border-t border-slate-800/60 flex items-center gap-2 
                       ${!sidebarOpen ? 'justify-center' : ''}`}>
        {backendReachable && !demoMode ? (
          <Wifi className="w-3.5 h-3.5 text-emerald-400 shrink-0" />
        ) : (
          <WifiOff className="w-3.5 h-3.5 text-amber-400 shrink-0" />
        )}
        {sidebarOpen && (
          <span className="text-[11px] text-slate-500">
            {backendReachable && !demoMode ? 'Connected to backend' : 'Demo mode'}
          </span>
        )}
      </div>
    </aside>
  );
};
