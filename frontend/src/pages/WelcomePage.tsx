/**
 * pages/WelcomePage.tsx — Landing/hero page shown to first-time users.
 */
import React from 'react';
import {
  Database, Shield, Zap, BarChart3, ArrowRight,
  MessageSquare, Code2, GitGraph, Lock, Star
} from 'lucide-react';
import { ThemeToggle } from '../components/ui/ThemeToggle';

interface Props {
  onLaunch: () => void;
  onDemo: () => void;
}

const FEATURES = [
  {
    icon: Shield,
    color: 'blue',
    title: '100% Private',
    desc: 'Runs entirely on your machine using Ollama. No cloud APIs, no data leaks — your data never leaves your computer.',
  },
  {
    icon: Zap,
    color: 'purple',
    title: 'Instant Insights',
    desc: 'Skip the pandas boilerplate. Ask "What\'s the top region by revenue?" and get charts, SQL, and answers in seconds.',
  },
  {
    icon: BarChart3,
    color: 'emerald',
    title: 'Beautiful Charts',
    desc: 'Auto-generated interactive Plotly charts, data tables, and exportable reports in CSV, Excel, PDF, and more.',
  },
  {
    icon: GitGraph,
    color: 'indigo',
    title: 'Multi-Agent Pipeline',
    desc: 'LangGraph orchestrates 12 specialized agents: supervisor, SQL planner, cross-checker, insight generator, and more.',
  },
  {
    icon: Code2,
    color: 'amber',
    title: 'SQL Transparency',
    desc: 'Every answer shows the exact SQL generated. Inspect, copy, or re-run with modifications — full auditability.',
  },
  {
    icon: Lock,
    color: 'rose',
    title: 'PII Protection',
    desc: 'Automatic PII detection and masking. Columns like email, phone, and SSN are never sent to the LLM.',
  },
];

const TECH_STACK = [
  'FastAPI', 'LangGraph', 'Ollama', 'MySQL', 'React', 'TypeScript', 'Plotly', 'SQLAlchemy',
];

const colorMap: Record<string, { bg: string; icon: string; border: string }> = {
  blue:    { bg: 'bg-blue-500/10',    icon: 'text-blue-400',    border: 'border-blue-500/20' },
  purple:  { bg: 'bg-purple-500/10',  icon: 'text-purple-400',  border: 'border-purple-500/20' },
  emerald: { bg: 'bg-emerald-500/10', icon: 'text-emerald-400', border: 'border-emerald-500/20' },
  indigo:  { bg: 'bg-indigo-500/10',  icon: 'text-indigo-400',  border: 'border-indigo-500/20' },
  amber:   { bg: 'bg-amber-500/10',   icon: 'text-amber-400',   border: 'border-amber-500/20' },
  rose:    { bg: 'bg-rose-500/10',    icon: 'text-rose-400',    border: 'border-rose-500/20' },
};

export const WelcomePage: React.FC<Props> = ({ onLaunch, onDemo }) => {
  return (
    <div className="min-h-screen hero-gradient grid-pattern text-slate-100 overflow-x-hidden">
      {/* Nav */}
      <nav className="fixed top-0 w-full z-50 border-b border-white/5 bg-black/20 backdrop-blur-xl">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-8 h-8 rounded-xl bg-indigo-600 flex items-center justify-center shadow-lg shadow-indigo-600/40">
              <Database className="w-4 h-4 text-white" />
            </div>
            <span className="text-lg font-bold tracking-tight">Quantai AI</span>
            <span className="text-xs text-slate-500 font-mono ml-1 hidden sm:block">V2.122205</span>
          </div>

          <div className="flex items-center gap-4">
            <ThemeToggle />
            <a href="#about" className="text-sm text-slate-400 hover:text-white transition-colors">
              About
            </a>
            <a
              href="https://github.com/dharaneeshiba02/Quantai-ai"
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm text-slate-400 hover:text-white transition-colors hidden sm:block"
            >
              GitHub
            </a>
            <button
              onClick={onDemo}
              className="btn-ghost text-sm"
            >
              <Star className="w-4 h-4 text-amber-400" />
              Live Demo
            </button>
            <button onClick={onLaunch} className="btn-primary text-sm">
              Launch App
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <main className="pt-40 pb-24 px-6">
        <div className="max-w-5xl mx-auto text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-indigo-500/10 
                         text-indigo-300 text-sm font-medium border border-indigo-500/20 mb-8 animate-fade-in">
            <span className="relative flex h-2 w-2">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75" />
              <span className="relative inline-flex rounded-full h-2 w-2 bg-indigo-500" />
            </span>
            Privacy-first · Local AI · No Cloud Required
          </div>

          {/* Title */}
          <h1 className="text-5xl md:text-7xl font-black tracking-tight leading-tight pb-4 animate-fade-in-up">
            <span className="glow-text">Talk to your data.</span>
            <br />
            <span className="text-slate-400 text-4xl md:text-6xl font-bold">Keep it private.</span>
          </h1>

          {/* Subtitle */}
          <p className="text-lg md:text-xl text-slate-400 leading-relaxed max-w-2xl mx-auto mt-6 mb-10 animate-fade-in-up animation-delay-200">
            Quantai AI is an MCP-powered local AI data analyst. Upload CSVs, connect databases, 
            ask questions in plain English — get back charts, SQL, and insights instantly.
            <strong className="text-slate-300"> Your data never leaves your machine.</strong>
          </p>

          {/* CTAs */}
          <div className="flex flex-col sm:flex-row items-center justify-center gap-4 animate-fade-in-up animation-delay-300">
            <button onClick={onLaunch} className="btn-primary px-8 py-4 text-base gap-2.5">
              <MessageSquare className="w-5 h-5" />
              Start Chatting with Data
              <ArrowRight className="w-4 h-4" />
            </button>
            <button
              onClick={() => document.getElementById('about')?.scrollIntoView({ behavior: 'smooth' })}
              className="flex items-center gap-2.5 px-8 py-4 rounded-xl font-medium text-slate-300 
                         bg-slate-800/50 hover:bg-slate-800 border border-slate-700 hover:border-slate-600 
                         transition-all text-base"
            >
              About Quantai
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>

          {/* Tech stack */}
          <div className="flex flex-wrap items-center justify-center gap-2 mt-12 animate-fade-in-up animation-delay-400">
            <span className="text-xs text-slate-600 mr-2">Built with</span>
            {TECH_STACK.map((tech) => (
              <span key={tech} className="px-2.5 py-1 rounded-lg bg-slate-800/60 text-slate-400 text-xs border border-slate-700/50 font-mono">
                {tech}
              </span>
            ))}
          </div>
        </div>

        {/* Architecture preview */}
        <div className="max-w-4xl mx-auto mt-24 animate-fade-in-up animation-delay-500">
          <p className="text-center text-xs text-slate-600 uppercase tracking-widest font-semibold mb-8">
            12-Node LangGraph Multi-Agent Pipeline
          </p>
          <div className="glass rounded-3xl p-8 overflow-x-auto">
            <div className="flex items-center gap-2 justify-center flex-wrap min-w-max mx-auto">
              {[
                { label: 'Supervisor', color: 'text-purple-400' },
                { label: '→' },
                { label: 'Source Resolver', color: 'text-blue-400' },
                { label: '→' },
                { label: 'Column Selector', color: 'text-cyan-400' },
                { label: '→' },
                { label: 'Query Planner', color: 'text-indigo-400' },
                { label: '→' },
                { label: 'SQL Compiler', color: 'text-violet-400' },
                { label: '→' },
                { label: 'Executor', color: 'text-emerald-400' },
              ].map((item, i) => (
                <span
                  key={i}
                  className={item.color
                    ? `px-3 py-1.5 rounded-lg text-sm font-semibold ${item.color} bg-current/5 border border-current/20`
                    : 'text-slate-700 text-lg'
                  }
                >
                  {item.label}
                </span>
              ))}
            </div>
            <div className="flex items-center gap-2 justify-center flex-wrap mt-3 min-w-max mx-auto">
              {[
                { label: '→' },
                { label: 'Cross-Check', color: 'text-amber-400' },
                { label: '→' },
                { label: 'Cleaner', color: 'text-rose-400' },
                { label: '→' },
                { label: 'Stats Analyzer', color: 'text-orange-400' },
                { label: '→' },
                { label: 'Visualizer', color: 'text-pink-400' },
                { label: '→' },
                { label: 'Insight', color: 'text-teal-400' },
                { label: '→' },
                { label: 'Confidence', color: 'text-lime-400' },
              ].map((item, i) => (
                <span
                  key={i}
                  className={item.color
                    ? `px-3 py-1.5 rounded-lg text-sm font-semibold ${item.color} bg-current/5 border border-current/20`
                    : 'text-slate-700 text-lg'
                  }
                >
                  {item.label}
                </span>
              ))}
            </div>
          </div>
        </div>

        {/* Features grid */}
        <div id="about" className="max-w-6xl mx-auto mt-24 grid md:grid-cols-2 lg:grid-cols-3 gap-5">
          {FEATURES.map(({ icon: Icon, color, title, desc }, i) => {
            const c = colorMap[color];
            return (
              <div
                key={title}
                className="glass-hover rounded-3xl p-7 animate-fade-in-up"
                style={{ animationDelay: `${i * 80}ms` }}
              >
                <div className={`w-12 h-12 rounded-2xl ${c.bg} ${c.border} border flex items-center justify-center mb-5`}>
                  <Icon className={`w-6 h-6 ${c.icon}`} />
                </div>
                <h3 className="text-lg font-bold text-white mb-2">{title}</h3>
                <p className="text-slate-400 text-sm leading-relaxed">{desc}</p>
              </div>
            );
          })}
        </div>

        {/* CTA bottom */}
        <div className="max-w-2xl mx-auto mt-24 text-center">
          <div className="glass rounded-3xl p-10">
            <h2 className="text-3xl font-bold text-white mb-3">Ready to analyze your data?</h2>
            <p className="text-slate-400 mb-8">
              Upload a CSV, ask a question, and get your first insight in under 30 seconds.
            </p>
            <div className="flex items-center justify-center gap-4">
              <button
                onClick={() => document.getElementById('about')?.scrollIntoView({ behavior: 'smooth' })}
                className="btn-primary px-8 py-3.5"
              >
                About Quantai
                <ArrowRight className="w-4 h-4" />
              </button>
              <button onClick={onDemo} className="btn-ghost">
                Try Demo First
              </button>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/60 py-8 px-6 text-center">
        <p className="text-slate-600 text-sm">
          Quantai AI — Local AI Data Analyst ·{' '}
          <a
            href="https://github.com/dharaneeshiba02/Quantai-ai"
            className="text-indigo-400 hover:text-indigo-300 transition-colors"
            target="_blank"
            rel="noopener noreferrer"
          >
            GitHub
          </a>
          {' '}· Built with ❤️ using FastAPI + LangGraph + React
        </p>
      </footer>
    </div>
  );
};
