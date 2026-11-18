/**
 * pages/panels/SettingsPanel.tsx — App settings display and model info.
 */
import React, { useCallback, useEffect, useState } from 'react';
import { Settings, Cpu, Shield, Zap, CheckCircle2, XCircle, RefreshCw, AlertTriangle } from 'lucide-react';
import { api } from '../../api/client';
import { useAppStore } from '../../store/useAppStore';

export const SettingsPanel: React.FC = () => {
  const [settings, setSettings] = useState<Record<string, any> | null>(null);
  const [health, setHealth] = useState<Record<string, any> | null>(null);
  const [healthError, setHealthError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);
  const [checkedAt, setCheckedAt] = useState<Date | null>(null);
  const { demoMode, backendReachable, setDemoMode } = useAppStore();

  const refreshStatus = useCallback(async () => {
    if (demoMode) return;
    setRefreshing(true);
    try {
      const [nextSettings, nextHealth] = await Promise.all([api.settings(), api.health()]);
      setSettings(nextSettings);
      setHealth(nextHealth);
      setHealthError(null);
      setCheckedAt(new Date());
    } catch (error) {
      setHealth(null);
      setHealthError(error instanceof Error ? error.message : 'Could not read system health.');
    } finally {
      setRefreshing(false);
    }
  }, [demoMode]);

  useEffect(() => {
    if (demoMode) {
      setHealth(null);
      setSettings(null);
      setHealthError(null);
      return;
    }
    void refreshStatus();
    const timer = window.setInterval(() => void refreshStatus(), 20000);
    return () => window.clearInterval(timer);
  }, [demoMode, backendReachable, refreshStatus]);

  const DEMO_SETTINGS = {
    llm_model: 'qwen3.5',
    llm_fallback_model: 'qwen2.5-coder:7b',
    llm_chat_model: 'qwen2.5-coder:7b',
    llm_chat_num_ctx: 4096,
    llm_chat_num_predict: 1024,
    llm_chat_timeout_sec: 120,
    llm_reasoning_enabled: false,
    llm_chat_keep_alive: '15m',
    accuracy_mode: 'balanced',
    schema_only_mode: false,
    pii_masking_enabled: true,
    column_selector_top_k: 20,
    sample_rows_for_llm: 3,
  };

  const s = settings || DEMO_SETTINGS;

  return (
    <div className="flex-1 overflow-y-auto p-6 max-w-4xl w-full">
      <div className="flex items-center justify-between gap-4 mb-6">
        <div className="flex items-center gap-2">
          <Settings className="w-5 h-5 text-indigo-400" />
          <h2 className="text-lg font-bold text-white">Settings</h2>
        </div>
        <span className="text-xs text-slate-500">{checkedAt ? `Updated ${checkedAt.toLocaleTimeString()}` : 'Health monitor'}</span>
      </div>

      {/* Demo mode toggle */}
      <div className="glass rounded-2xl p-5 mb-4">
        <div className="flex items-center justify-between">
          <div>
            <p className="text-sm font-semibold text-white">Demo Mode</p>
            <p className="text-xs text-slate-500 mt-0.5">Use mock data — no backend required</p>
          </div>
          <button
            onClick={() => setDemoMode(!demoMode)}
            className={`relative w-11 h-6 rounded-full transition-colors ${demoMode ? 'bg-indigo-600' : 'bg-slate-700'}`}
          >
            <div className={`absolute top-1 left-1 w-4 h-4 rounded-full bg-white transition-transform ${demoMode ? 'translate-x-5' : ''}`} />
          </button>
        </div>
      </div>

      {/* System health */}
      <div className="glass rounded-2xl p-5 mb-4">
        <div className="flex items-center justify-between gap-3 mb-4">
          <div className="flex items-center gap-2">
            <Zap className="w-4 h-4 text-indigo-400" />
            <div>
              <p className="text-sm font-semibold text-white">Component Health</p>
              <p className="text-[11px] text-slate-500 mt-0.5">Backend, local models, storage, database, and analysis tools</p>
            </div>
          </div>
          <button
            onClick={() => void refreshStatus()}
            disabled={refreshing || demoMode}
            className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 px-3 py-2 text-xs text-slate-300 transition hover:bg-white/5 disabled:opacity-50"
            title="Refresh component status"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
        {healthError && (
          <div role="status" className="mb-4 flex items-start gap-2 rounded-xl border border-amber-400/20 bg-amber-400/5 px-3 py-2.5 text-xs text-amber-200">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />{healthError}
          </div>
        )}
        <div className="grid gap-2 sm:grid-cols-2">
          {(health?.components || [{ name: 'Backend API', ok: backendReachable, detail: backendReachable ? 'Connected' : 'Backend is unavailable', fix: backendReachable ? null : 'Start the local app server.' }]).map((component: any) => (
            <div key={component.name} className="flex min-w-0 items-start gap-3 rounded-xl border border-white/[0.06] bg-black/[0.08] px-3.5 py-3">
              {component.ok
                ? <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
                : <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-400" />}
              <div className="min-w-0 flex-1">
                <p className="text-xs font-medium text-slate-200">{component.name}</p>
                <p className="mt-0.5 break-words text-[11px] leading-4 text-slate-500">{component.detail}</p>
                {component.fix && <p className="mt-1 break-words text-[11px] leading-4 text-amber-300">{component.fix}</p>}
              </div>
            </div>
          ))}
        </div>
        {health?.error && (
          <div className="mt-4 rounded-xl border border-amber-500/20 bg-amber-500/10 px-3 py-2.5">
            <p className="text-xs text-amber-200">{health.error}</p>
            {health.fix && <pre className="mt-2 whitespace-pre-wrap text-[11px] text-slate-400">{health.fix}</pre>}
          </div>
        )}
      </div>

      {/* LLM Settings */}
      <div className="glass rounded-2xl p-5 mb-4">
        <div className="flex items-center gap-2 mb-4">
          <Cpu className="w-4 h-4 text-purple-400" />
          <p className="text-sm font-semibold text-white">LLM Configuration</p>
        </div>
        <div className="space-y-3">
          {[
            { label: 'Primary Model', value: s.llm_model },
            { label: 'Default Chat Model', value: s.llm_chat_model },
            { label: 'Reasoning Mode', value: s.llm_reasoning_enabled ? 'Enabled' : 'Fast (disabled)' },
            { label: 'Chat Context', value: `${s.llm_chat_num_ctx} tokens` },
            { label: 'Chat Response Limit', value: `${s.llm_chat_num_predict} tokens` },
            { label: 'Chat Timeout', value: `${s.llm_chat_timeout_sec}s` },
            { label: 'Keep Models Loaded', value: s.llm_chat_keep_alive },
            { label: 'Recovery Model', value: s.llm_fallback_model },
            { label: 'Accuracy Mode', value: s.accuracy_mode },
            { label: 'Top-K Columns', value: s.column_selector_top_k },
            { label: 'Sample Rows', value: s.sample_rows_for_llm },
          ].map(({ label, value }) => (
            <div key={label} className="flex items-center justify-between">
              <span className="text-sm text-slate-500">{label}</span>
              <span className="text-sm text-slate-200 font-mono">{String(value)}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Privacy */}
      <div className="glass rounded-2xl p-5">
        <div className="flex items-center gap-2 mb-4">
          <Shield className="w-4 h-4 text-blue-400" />
          <p className="text-sm font-semibold text-white">Privacy & Security</p>
        </div>
        <div className="space-y-3">
          {[
            { label: 'Schema-only mode', value: s.schema_only_mode ? 'Enabled' : 'Disabled', ok: s.schema_only_mode },
            { label: 'PII masking', value: s.pii_masking_enabled ? 'Enabled' : 'Disabled', ok: s.pii_masking_enabled },
            { label: 'Cloud telemetry', value: 'Disabled', ok: true },
            { label: 'Data stays local', value: 'Always', ok: true },
          ].map(({ label, value, ok }) => (
            <div key={label} className="flex items-center justify-between">
              <span className="text-sm text-slate-500">{label}</span>
              <span className={`text-sm font-medium ${ok ? 'text-emerald-400' : 'text-slate-400'}`}>{value}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
