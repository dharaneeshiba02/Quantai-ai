/**
 * components/data/UploadDialog.tsx — Drag-and-drop CSV/Excel upload with progress.
 */
import React, { useCallback, useState } from 'react';
import { Upload, X, FileSpreadsheet, CheckCircle2, AlertTriangle, Loader2 } from 'lucide-react';
import { api } from '../../api/client';
import { useAppStore } from '../../store/useAppStore';

interface Props {
  onClose: () => void;
}

export const UploadDialog: React.FC<Props> = ({ onClose }) => {
  const [dragging, setDragging] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<'idle' | 'uploading' | 'done' | 'error'>('idle');
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  const {
    addSource, clearMessages, setDocumentMode,
    setSelectedSourceId, setUploadDialogOpen,
  } = useAppStore();

  const handleFile = useCallback((f: File) => {
    const allowed = ['.csv', '.xlsx', '.xls'];
    const ext = '.' + f.name.split('.').pop()?.toLowerCase();
    if (!allowed.includes(ext)) {
      setError('Upload a CSV or Excel dataset.');
      return;
    }
    const maxSize = 200;
    if (f.size > maxSize * 1024 * 1024) {
      setError(`File too large. Maximum size is ${maxSize} MB for this file type.`);
      return;
    }
    setFile(f);
    setError(null);
    setStatus('idle');
    setResult(null);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files[0];
    if (f) handleFile(f);
  }, [handleFile]);

  const handleUpload = async () => {
    if (!file) return;
    setStatus('uploading');
    setProgress(0);
    setError(null);

    try {
      const ext = '.' + file.name.split('.').pop()?.toLowerCase();
      const res = await api.upload(file, setProgress);
      setResult({ kind: 'dataset', ...res });
      setDocumentMode(false);
      clearMessages();
      addSource({
        id: res.source_id,
        name: file.name,
        source_type: ext === '.csv' ? 'csv' : 'excel',
        table_name: res.table_name,
        row_count: res.row_count,
        col_count: res.col_count,
        created_at: new Date().toISOString(),
      });
      setSelectedSourceId(res.source_id);
      setStatus('done');

      // Auto-close after 2s
      setTimeout(() => {
        setUploadDialogOpen(false);
      }, 2000);
    } catch (err: any) {
      setError(err.message || 'Upload failed');
      setStatus('error');
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

      {/* Dialog */}
      <div className="relative w-full max-w-lg glass rounded-3xl p-8 shadow-2xl animate-scale-in">
        {/* Close */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-2 rounded-xl text-slate-500 hover:text-slate-200 hover:bg-slate-800 transition-colors"
        >
          <X className="w-5 h-5" />
        </button>

        <h2 className="text-xl font-bold text-white mb-1">Upload Data</h2>
        <p className="text-sm text-slate-400 mb-6">
          CSV and Excel datasets up to 200 MB. Files stay on your machine.
        </p>

        {status === 'done' ? (
          <div className="flex flex-col items-center gap-4 py-8 text-center">
            <div className="w-16 h-16 rounded-full bg-emerald-500/15 flex items-center justify-center">
              <CheckCircle2 className="w-8 h-8 text-emerald-400" />
            </div>
            <div>
              <p className="text-white font-semibold text-lg">Dataset imported</p>
              <p className="text-slate-400 text-sm mt-1">{result?.row_count?.toLocaleString()} rows × {result?.col_count} columns imported in {result?.elapsed_sec?.toFixed(1)}s</p>
            </div>
          </div>
        ) : (
          <>
            {/* Drop zone */}
            <div
              onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              onClick={() => document.getElementById('file-input')?.click()}
              className={`relative flex flex-col items-center justify-center gap-4 p-10 rounded-2xl 
                         border-2 border-dashed cursor-pointer transition-all select-none
                         ${dragging
                           ? 'border-indigo-400 bg-indigo-500/10 scale-[1.01]'
                           : file
                           ? 'border-emerald-500/50 bg-emerald-500/5'
                           : 'border-slate-700 hover:border-slate-600 bg-slate-900/40'
                         }`}
            >
              <input
                id="file-input"
                type="file"
                accept=".csv,.xlsx,.xls"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              />

              {file ? (
                <>
                  <div className="w-14 h-14 rounded-2xl bg-emerald-500/15 flex items-center justify-center">
                    <FileSpreadsheet className="w-7 h-7 text-emerald-400" />
                  </div>
                  <div className="text-center">
                    <p className="text-white font-medium">{file.name}</p>
                    <p className="text-slate-400 text-sm">{formatSize(file.size)}</p>
                  </div>
                </>
              ) : (
                <>
                  <div className="w-14 h-14 rounded-2xl bg-indigo-500/10 flex items-center justify-center">
                    <Upload className="w-7 h-7 text-indigo-400" />
                  </div>
                  <div className="text-center">
                    <p className="text-white font-medium">Drop file here or click to browse</p>
                    <p className="text-slate-500 text-sm">.csv, .xlsx, .xls</p>
                  </div>
                </>
              )}
            </div>

            {/* Upload progress */}
            {status === 'uploading' && (
              <div className="mt-4">
                <div className="flex items-center justify-between text-xs text-slate-400 mb-1.5">
                  <span className="flex items-center gap-1.5">
                    <Loader2 className="w-3.5 h-3.5 animate-spin text-indigo-400" />
                    Uploading & importing dataset…
                  </span>
                  <span>{progress}%</span>
                </div>
                <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-indigo-500 transition-all duration-300 rounded-full"
                    style={{ width: `${progress}%` }}
                  />
                </div>
              </div>
            )}

            {/* Error */}
            {error && (
              <div className="mt-4 flex items-start gap-2.5 px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/20">
                <AlertTriangle className="w-4 h-4 text-red-400 shrink-0 mt-0.5" />
                <p className="text-red-300 text-sm">{error}</p>
              </div>
            )}

            {/* Actions */}
            <div className="flex gap-3 mt-6">
              <button onClick={onClose} className="btn-ghost flex-1 justify-center">
                Cancel
              </button>
              <button
                onClick={handleUpload}
                disabled={!file || status === 'uploading'}
                className="btn-primary flex-1 justify-center"
              >
                {status === 'uploading' ? (
                  <><Loader2 className="w-4 h-4 animate-spin" /> Importing…</>
                ) : (
                  <><Upload className="w-4 h-4" /> Upload & Import</>
                )}
              </button>
            </div>
          </>
        )}
      </div>
    </div>
  );
};
