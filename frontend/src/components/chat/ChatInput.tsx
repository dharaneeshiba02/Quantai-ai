/**
 * components/chat/ChatInput.tsx — Query input with suggested questions.
 */
import React, { useState, useRef, useEffect } from 'react';
import { Cpu, Send, Square, Sparkles } from 'lucide-react';
import type { ChatModel } from '../../api/client';

interface Props {
  onSubmit: (question: string) => void;
  isLoading: boolean;
  onCancel: () => void;
  suggestions?: string[];
  disabled?: boolean;
  models?: ChatModel[];
  selectedModel?: string;
  onModelChange?: (model: string) => void;
  modelError?: string | null;
}

export const ChatInput: React.FC<Props> = ({
  onSubmit, isLoading, onCancel, suggestions = [], disabled,
  models = [], selectedModel, onModelChange, modelError,
}) => {
  const [value, setValue] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 160)}px`;
    }
  }, [value]);

  const handleSubmit = () => {
    const q = value.trim();
    if (!q || isLoading || disabled) return;
    setValue('');
    onSubmit(q);
  };

  const handleKey = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSuggestion = (q: string) => {
    if (isLoading || disabled) return;
    setValue(q);
    textareaRef.current?.focus();
  };

  return (
    <div className="flex flex-col gap-3">
      {onModelChange && (
        <div className="flex items-center gap-2 self-start text-xs text-slate-400">
          <Cpu className="h-3.5 w-3.5 text-indigo-300" />
          <label htmlFor="chat-model">Model</label>
          <select
            id="chat-model"
            value={selectedModel || ''}
            onChange={(event) => onModelChange(event.target.value)}
            disabled={isLoading || models.length === 0}
            className="chat-model-select rounded-lg border border-slate-200 px-2.5 py-1.5 text-xs outline-none focus:border-indigo-300 disabled:opacity-50"
            title={modelError || 'Choose the local Ollama model for this chat'}
          >
            {models.map((model) => (
              <option key={model.id} value={model.id} disabled={!model.available}>
                {model.label}{model.available ? '' : ' (not installed)'}
              </option>
            ))}
            {models.length === 0 && <option value="">No Ollama models available</option>}
          </select>
          {modelError && <span role="status" className="text-amber-300">{modelError}</span>}
        </div>
      )}
      {/* Suggested questions */}
      {suggestions.length > 0 && !isLoading && (
        <div className="flex flex-wrap gap-2">
          {suggestions.slice(0, 4).map((q, i) => (
            <button
              key={i}
              onClick={() => handleSuggestion(q)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs text-slate-300 
                         bg-slate-800/60 hover:bg-slate-700/80 border border-slate-700/50 
                         hover:border-indigo-500/40 transition-all group"
            >
              <Sparkles className="w-3 h-3 text-indigo-400 group-hover:text-indigo-300 transition-colors shrink-0" />
              <span className="text-left line-clamp-1">{q}</span>
            </button>
          ))}
        </div>
      )}

      {/* Input area */}
      <div className="relative flex items-end gap-3 glass rounded-2xl p-3 focus-within:border-indigo-500/40 transition-all">
        <textarea
          ref={textareaRef}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKey}
          placeholder={disabled ? 'Select a data source first…' : 'Ask a question about your data…'}
          disabled={disabled || isLoading}
          rows={1}
          className="flex-1 bg-transparent resize-none outline-none text-sm text-slate-200 
                     placeholder:text-slate-600 leading-relaxed disabled:opacity-50 
                     min-h-[36px] max-h-[160px]"
          style={{ scrollbarWidth: 'none' }}
        />

        {isLoading ? (
          <button
            onClick={onCancel}
            className="shrink-0 w-9 h-9 rounded-xl bg-red-500/20 hover:bg-red-500/30 
                       border border-red-500/30 flex items-center justify-center 
                       text-red-400 transition-all"
            title="Stop generating"
          >
            <Square className="w-4 h-4 fill-current" />
          </button>
        ) : (
          <button
            onClick={handleSubmit}
            disabled={!value.trim() || disabled}
            className="btn-primary shrink-0 w-9 h-9 !px-0 !py-0 justify-center"
            title="Send (Enter)"
          >
            <Send className="w-4 h-4" />
          </button>
        )}
      </div>
      <p className="text-center text-[10px] text-slate-700">
        Press <kbd className="font-mono bg-slate-800 px-1 rounded text-slate-500">Enter</kbd> to send · 
        <kbd className="font-mono bg-slate-800 px-1 rounded text-slate-500 mx-1">Shift+Enter</kbd> for new line
      </p>
    </div>
  );
};
