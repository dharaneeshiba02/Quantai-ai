/**
 * pages/panels/SettingsPanel.tsx — App settings display and model info.
 */
import React, { useEffect, useState } from 'react';
import { Settings, Cpu, Shield, Zap, CheckCircle2, XCircle } from 'lucide-react';
import { api } from '../../api/client';
import { useAppStore } from '../../store/useAppStore';

export const SettingsPanel: React.FC = () => {
  const [settings, setSettings] = useState<Record<string, any> | null>(null);
  const [health, setHealth] = useState<Record<string, any> | null>(null);
  const { demoMode, backendReachable, setDemoMode } = useAppStore();

  useEffect(() => {
    if (demoMode || !backendReachable) return;
    Promise.all([api.settings(), api.health()])
      .then(([s, h]) => { setSettings(s); setHealth(h); })
      .catch(() => {});
  }, [demoMode, backendReachable]);

  const DEMO_SETTINGS = {
    llm_model: 'qwen2.5-coder:7b',
    llm_fallback_model: 'qwen2.5-coder:3b',
    accuracy_mode: 'balanced',
    schema_only_mode: false,
    pii_masking_enabled: true,
    column_selector_top_k: 20,
    sample_rows_for_llm: 3,
  };

  const s = settings || DEMO_SETTINGS;

  return (
    <div className="flex-1 overflow-y-auto p-6 max-w-2xl">
      <div className="flex items-center gap-2 mb-6">
        <Settings className="w-5 h-5 text-indigo-400" />
        <h2 className="text-lg font-bold text-white">Settings</h2>
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
        <div className="flex items-center gap-2 mb-4">
          <Zap className="w-4 h-4 text-indigo-400" />
          <p className="text-sm font-semibold text-white">System Status</p>
        </div>
        <div className="space-y-3">
          {[
            { label: 'Backend API', ok: backendReachable },
            { label: 'LLM Provider', ok: health?.provider_reachable ?? demoMode },
            { label: 'LLM Model', ok: health?.model_available ?? demoMode, detail: health?.model },
            { label: 'Database', ok: health?.database?.status === 'ok' || demoMode },
          ].map(({ label, ok, detail }) => (
            <div key={label} className="flex items-center justify-between">
              <span className="text-sm text-slate-400">{label}</span>
              <div className="flex items-center gap-2">
                {detail && <span className="text-xs font-mono text-slate-500">{detail}</span>}
                {ok
                  ? <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  : <XCircle className="w-4 h-4 text-red-400" />
                }
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
            { label: 'Fallback Model', value: s.llm_fallback_model },
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
