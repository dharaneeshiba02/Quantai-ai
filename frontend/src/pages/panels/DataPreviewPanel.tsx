/**
 * pages/panels/DataPreviewPanel.tsx — Shows source schema, stats, and preview rows.
 */
import React, { useEffect, useState } from 'react';
import { Database, BarChart3, Calendar, Eye, AlertTriangle } from 'lucide-react';
import { DataTable } from '../../components/tables/DataTable';
import { api } from '../../api/client';
import { useAppStore } from '../../store/useAppStore';
import { DEMO_PREVIEW_ROWS } from '../../api/demo';
import type { ColumnCatalogEntry } from '../../api/client';

interface Props {
  sourceId: string | null;
}

function typeColor(type: string) {
  switch (type) {
    case 'int': return 'text-sky-400 bg-sky-500/10 border-sky-500/20';
    case 'float': return 'text-blue-400 bg-blue-500/10 border-blue-500/20';
    case 'str': return 'text-purple-400 bg-purple-500/10 border-purple-500/20';
    case 'date':
    case 'datetime': return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20';
    case 'bool': return 'text-amber-400 bg-amber-500/10 border-amber-500/20';
    default: return 'text-slate-400 bg-slate-800 border-slate-700';
  }
}

export const DataPreviewPanel: React.FC<Props> = ({ sourceId }) => {
  const [catalog, setCatalog] = useState<ColumnCatalogEntry[]>([]);
  const [previewRows, setPreviewRows] = useState<Record<string, any>[]>([]);
  const [activeTab, setActiveTab] = useState<'schema' | 'preview'>('schema');
  const [loading, setLoading] = useState(false);
  const [catalogError, setCatalogError] = useState<string | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const { sourceOverview, demoMode, backendReachable } = useAppStore();

  useEffect(() => {
    let active = true;
    setCatalog([]);
    setPreviewRows([]);
    setCatalogError(null);
    setPreviewError(null);
    if (!sourceId) {
      setLoading(false);
      return () => { active = false; };
    }
    setLoading(true);
    if (demoMode) {
      setPreviewRows(DEMO_PREVIEW_ROWS);
      setLoading(false);
      return () => { active = false; };
    }
    if (!backendReachable) {
      setCatalogError('Backend is disconnected. Reconnect to load this dataset schema.');
      setPreviewError('Backend is disconnected. Reconnect to load this dataset preview.');
      setLoading(false);
      return () => { active = false; };
    }

    Promise.allSettled([
      api.sources.catalog(sourceId),
      api.sources.preview(sourceId),
    ]).then(([catalogResult, previewResult]) => {
      if (!active) return;
      if (catalogResult.status === 'fulfilled') setCatalog(catalogResult.value);
      else setCatalogError(catalogResult.reason instanceof Error ? catalogResult.reason.message : 'Could not load this dataset schema.');
      if (previewResult.status === 'fulfilled') setPreviewRows(previewResult.value.rows);
      else setPreviewError(previewResult.reason instanceof Error ? previewResult.reason.message : 'Could not load this dataset preview.');
    }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [sourceId, demoMode, backendReachable]);

  if (!sourceId) {
    return (
      <div className="flex-1 flex items-center justify-center text-center p-8">
        <div>
          <Database className="w-12 h-12 text-slate-700 mx-auto mb-4" />
          <p className="text-slate-500">Select a data source to view its schema and data.</p>
        </div>
      </div>
    );
  }

  const previewColumns = previewRows.length > 0 ? Object.keys(previewRows[0]) : [];

  return (
    <div className="flex-1 overflow-y-auto p-6">
      {/* Overview cards */}
      {sourceOverview && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          {[
            { label: 'Total Rows', value: sourceOverview.row_count?.toLocaleString(), icon: BarChart3, color: 'text-indigo-400' },
            { label: 'Columns', value: sourceOverview.col_count, icon: Database, color: 'text-purple-400' },
            { label: 'Date Columns', value: sourceOverview.date_columns?.length || 0, icon: Calendar, color: 'text-emerald-400' },
            { label: 'High Null Cols', value: sourceOverview.null_high_columns?.length || 0, icon: Eye, color: 'text-amber-400' },
          ].map(({ label, value, icon: Icon, color }) => (
            <div key={label} className="glass rounded-2xl p-4">
              <div className="flex items-center gap-2 mb-2">
                <Icon className={`w-4 h-4 ${color}`} />
                <span className="text-xs text-slate-500">{label}</span>
              </div>
              <p className="text-2xl font-bold text-white">{value}</p>
            </div>
          ))}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-1 p-1 glass rounded-xl w-fit mb-6">
        {(['schema', 'preview'] as const).map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-all capitalize
              ${activeTab === tab ? 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30' : 'text-slate-500 hover:text-slate-300'}`}
          >
            {tab}
          </button>
        ))}
      </div>

      {activeTab === 'schema' && (
        <div className="space-y-2">
          {loading ? (
            Array.from({ length: 8 }).map((_, i) => (
              <div key={i} className="h-12 shimmer rounded-xl" />
            ))
          ) : catalog.length > 0 ? (
            catalog.map((col) => (
              <div key={col.safe_name} className="glass rounded-xl px-4 py-3 flex items-center gap-4">
                <span className={`px-2 py-0.5 rounded text-[10px] font-semibold border ${typeColor(col.col_type)}`}>
                  {col.col_type}
                </span>
                <span className="font-mono text-sm text-slate-200 min-w-[140px]">{col.safe_name}</span>
                {col.original_name !== col.safe_name && (
                  <span className="text-xs text-slate-600">({col.original_name})</span>
                )}
                <div className="flex-1 flex items-center gap-4 text-xs text-slate-500">
                  {col.null_pct != null && (
                    <span className={col.null_pct > 20 ? 'text-amber-400' : ''}>
                      {col.null_pct.toFixed(1)}% null
                    </span>
                  )}
                  {col.distinct_count != null && <span>{col.distinct_count.toLocaleString()} distinct</span>}
                  {col.col_min && col.col_max && (
                    <span>{col.col_min} – {col.col_max}</span>
                  )}
                </div>
                <div className="flex gap-1.5">
                  {col.is_date_col && <span className="px-1.5 py-0.5 rounded text-[9px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">DATE</span>}
                  {col.is_key_col && <span className="px-1.5 py-0.5 rounded text-[9px] bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">KEY</span>}
                  {col.is_pii && <span className="px-1.5 py-0.5 rounded text-[9px] bg-red-500/10 text-red-400 border border-red-500/20">PII</span>}
                </div>
              </div>
            ))
          ) : (
            <div role={catalogError ? 'alert' : 'status'} className="data-panel-error mx-auto flex max-w-xl items-start gap-3 rounded-xl border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-sm text-amber-700">
              <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
              <span>{catalogError || (demoMode ? 'Schema details are not included in demo data.' : 'No schema details are available for this dataset.')}</span>
            </div>
          )}
        </div>
      )}

      {activeTab === 'preview' && (
        loading ? (
          <div className="h-80 shimmer rounded-xl" />
        ) : previewError ? (
          <div role="alert" className="data-panel-error mx-auto flex max-w-xl items-start gap-3 rounded-xl border border-amber-500/20 bg-amber-500/5 px-4 py-3 text-sm text-amber-700">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
            <span>{previewError}</span>
          </div>
        ) : (
          <DataTable rows={previewRows} columns={previewColumns} />
        )
      )}
    </div>
  );
};
