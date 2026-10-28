"""
tests/test_chatbot_general.py — Tests for general-purpose chatbot engine and session memory.
"""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from app.rag.chatbot import (
    CustomChatbotChain,
    clear_session_history,
    get_chatbot_chain,
    get_session_history,
)


class ChatbotGeneralTests(unittest.TestCase):
    def setUp(self):
        clear_session_history("session_alpha")
        clear_session_history("session_beta")
        self.vector_patcher = patch("app.rag.chatbot.get_vectorstore", return_value=None)
        self.vector_patcher.start()

    def tearDown(self):
        self.vector_patcher.stop()
        clear_session_history("session_alpha")
        clear_session_history("session_beta")

    def test_session_isolation(self):
        """Messages in session_alpha must not leak into session_beta."""
        chain = get_chatbot_chain()

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Alpha response")

        with patch("app.rag.chatbot.get_llm", return_value=mock_llm):
            res_a = chain.invoke({"question": "What is our 2025 revenue?", "session_id": "session_alpha"})
            self.assertEqual(res_a["answer"], "Alpha response")

        # Verify session_alpha has the question and answer
        hist_a = get_session_history("session_alpha")
        self.assertEqual(len(hist_a), 2)
        self.assertEqual(hist_a[0]["content"], "What is our 2025 revenue?")

        # Verify session_beta is completely empty
        hist_b = get_session_history("session_beta")
        self.assertEqual(len(hist_b), 0)

    def test_clear_session_history(self):
        """Clearing one session history leaves other sessions untouched."""
        chain = get_chatbot_chain()
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Mock response")

        with patch("app.rag.chatbot.get_llm", return_value=mock_llm):
            chain.invoke({"question": "Question A", "session_id": "session_alpha"})
            chain.invoke({"question": "Question B", "session_id": "session_beta"})

        self.assertEqual(len(get_session_history("session_alpha")), 2)
        self.assertEqual(len(get_session_history("session_beta")), 2)

        clear_session_history("session_alpha")
        self.assertEqual(len(get_session_history("session_alpha")), 0)
        self.assertEqual(len(get_session_history("session_beta")), 2)

    def test_empty_question_handling(self):
        chain = get_chatbot_chain()
        res = chain.invoke({"question": "   ", "session_id": "session_alpha"})
        self.assertIn("Please ask a question", res["answer"])

    def test_error_handling_when_llm_fails(self):
        chain = get_chatbot_chain()
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = RuntimeError("Service connection failed")

        with patch("app.rag.chatbot.get_llm", return_value=mock_llm):
            res = chain.invoke({"question": "What is the average?", "session_id": "session_alpha"})
            self.assertIn("could not complete the query", res["answer"])
            self.assertIn("Service connection failed", res["answer"])


if __name__ == "__main__":
    unittest.main()
