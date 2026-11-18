import React, { useState } from 'react';
import { Download, FileText, Loader2, Sparkles, X } from 'lucide-react';
import { api } from '../../api/client';

interface Props {
  sourceId: string;
  sourceName: string;
  onClose: () => void;
}

export const DatasetReportDialog: React.FC<Props> = ({ sourceId, sourceName, onClose }) => {
  const [title, setTitle] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    const cleanedTitle = title.trim();
    if (!cleanedTitle) {
      setError('Enter a title for your report.');
      return;
    }

    setLoading(true);
    setError(null);
    try {
      await api.sources.report(sourceId, cleanedTitle);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Could not create the PDF. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center p-4" role="presentation">
      <button
        type="button"
        aria-label="Close report dialog"
        className="absolute inset-0 bg-black/70 backdrop-blur-sm"
        onClick={onClose}
      />
      <form
        onSubmit={handleSubmit}
        className="relative w-full max-w-md rounded-3xl border border-white/10 bg-[#111722] p-6 sm:p-7 shadow-[0_30px_100px_rgba(0,0,0,.55)] animate-scale-in"
      >
        <button
          type="button"
          onClick={onClose}
          className="absolute right-4 top-4 rounded-lg p-2 text-slate-500 transition hover:bg-white/5 hover:text-white"
          aria-label="Close"
        >
          <X className="h-4 w-4" />
        </button>

        <div className="mb-5 flex h-12 w-12 items-center justify-center rounded-2xl border border-indigo-300/15 bg-indigo-400/10">
          <FileText className="h-5 w-5 text-indigo-300" />
        </div>
        <p className="mb-2 flex items-center gap-2 text-[10px] font-semibold uppercase tracking-[0.18em] text-indigo-300">
          <Sparkles className="h-3.5 w-3.5" /> Dataset intelligence report
        </p>
        <h2 className="text-xl font-semibold tracking-tight text-white">Name your PDF</h2>
        <p className="mt-2 text-sm leading-6 text-slate-400">
          Create a detailed report for <span className="font-medium text-slate-200">{sourceName}</span>, with charts when the data supports them.
        </p>

        <label htmlFor="dataset-report-title" className="mt-6 block text-xs font-medium text-slate-300">
          Report title
        </label>
        <div className="relative mt-2">
          <input
            id="dataset-report-title"
            autoFocus
            value={title}
            onChange={(event) => { setTitle(event.target.value.slice(0, 25)); setError(null); }}
            maxLength={25}
            placeholder="e.g. Sales performance"
            className="w-full rounded-xl border border-white/10 bg-[#090d15] px-4 py-3 pr-16 text-sm text-white outline-none transition placeholder:text-slate-600 focus:border-indigo-400/50 focus:ring-2 focus:ring-indigo-400/10"
          />
          <span className="absolute right-3 top-1/2 -translate-y-1/2 text-[11px] tabular-nums text-slate-600">
            {title.length}/25
          </span>
        </div>
        <p className="mt-2 text-[11px] text-slate-500">Large datasets receive a deeper 6-8 page report. On slower laptops, local Qwen3.5 may take a few minutes before the PDF downloads.</p>

        {error && <p role="alert" className="mt-4 rounded-xl border border-red-400/20 bg-red-400/10 px-3 py-2 text-xs text-red-300">{error}</p>}

        <div className="mt-6 flex items-center justify-between border-t border-white/[0.07] pt-5">
          <span className="text-[11px] text-slate-600">Prepared locally - Quantai AI</span>
          <button type="submit" disabled={loading || !title.trim()} className="btn-primary rounded-xl px-4 py-2.5 text-sm disabled:cursor-not-allowed disabled:opacity-50">
            {loading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Download className="h-4 w-4" />}
            {loading ? 'Running Qwen3.5 + building PDF...' : 'Download PDF'}
          </button>
        </div>
      </form>
    </div>
  );
};
