import { useCallback, useEffect, useState } from 'react';
import { WelcomePage } from './pages/WelcomePage';
import { ChatPage } from './pages/ChatPage';
import { isDemoMode } from './api/demo';
import { api } from './api/client';
import { useAppStore } from './store/useAppStore';

function App() {
  const [currentPage, setCurrentPage] = useState<'welcome' | 'chat'>(() =>
    window.location.hash === '#chat' || window.location.hash === '#demo' ? 'chat' : 'welcome',
  );
  const setBackendReachable = useAppStore((state) => state.setBackendReachable);
  const setDemoMode = useAppStore((state) => state.setDemoMode);
  const startNewChat = useAppStore((state) => state.startNewChat);

  const navigate = useCallback((page: 'welcome' | 'chat', demo = false) => {
    const nextHash = page === 'chat' ? (demo ? '#demo' : '#chat') : '';
    if (window.location.hash !== nextHash) {
      window.history.pushState(null, '', `${window.location.pathname}${window.location.search}${nextHash}`);
    }
    setCurrentPage(page);
  }, []);

  useEffect(() => {
    if (isDemoMode() || window.location.hash === '#demo') setDemoMode(true);

    const syncPageFromHash = () => {
      setCurrentPage(window.location.hash === '#chat' || window.location.hash === '#demo' ? 'chat' : 'welcome');
      if (window.location.hash === '#demo') setDemoMode(true);
    };
    window.addEventListener('hashchange', syncPageFromHash);

    let active = true;
    let retryTimer: number | undefined;
    const checkBackend = () => {
      api.health()
        .then(() => { if (active) setBackendReachable(true); })
        .catch(() => { if (active) setBackendReachable(false); })
        .finally(() => {
          if (active) retryTimer = window.setTimeout(checkBackend, 15000);
        });
    };
    checkBackend();

    return () => {
      active = false;
      window.removeEventListener('hashchange', syncPageFromHash);
      if (retryTimer !== undefined) window.clearTimeout(retryTimer);
    };
  }, [setBackendReachable, setDemoMode]);

  if (currentPage === 'welcome') {
    return (
      <WelcomePage
        onLaunch={() => { setDemoMode(false); startNewChat(null); navigate('chat'); }}
        onDemo={() => {
          setDemoMode(true);
          navigate('chat', true);
        }}
      />
    );
  }

  return <ChatPage onNavigate={navigate} />;
}

export default App;
