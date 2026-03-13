"""
In-memory session storage for interview state. Can be replaced with Redis later.
"""

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)

_sessions: dict[str, dict[str, Any]] = {}


def create_session(parsed_jd: dict[str, Any] | None = None) -> str:
    """Create a new session and return session_id."""
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "parsed_jd": parsed_jd or {},
        "questions": [],
        "history": [],
        "current_index": 0,
    }
    return session_id


def get_session(session_id: str) -> dict[str, Any] | None:
    """Return session data or None."""
    return _sessions.get(session_id)


def set_questions(session_id: str, questions: list[str]) -> None:
    """Set the question list for the session."""
    if session_id in _sessions:
        _sessions[session_id]["questions"] = list(questions)


def append_turn(
    session_id: str,
    question: str,
    answer: str,
    score: int | None = None,
    feedback: str = "",
    followup: str = "",
) -> None:
    """Append a Q&A turn to session history."""
    if session_id not in _sessions:
        return
    _sessions[session_id]["history"].append({
        "question": question,
        "answer": answer,
        "score": score,
        "feedback": feedback,
        "followup": followup,
    })
    _sessions[session_id]["current_index"] = len(_sessions[session_id]["history"])


def get_history(session_id: str) -> list[dict[str, Any]]:
    """Return session history for final report."""
    s = _sessions.get(session_id)
    if not s:
        return []
    return list(s.get("history", []))


def get_current_question(session_id: str) -> str | None:
    """Return the current question index and text."""
    s = _sessions.get(session_id)
    if not s:
        return None
    questions = s.get("questions", [])
    idx = len(s.get("history", []))
    if 0 <= idx < len(questions):
        return questions[idx]
    return None


def session_finished(session_id: str, question_limit: int = 5) -> bool:
    """Return True if the session has reached the question limit."""
    s = _sessions.get(session_id)
    if not s:
        return True
    return len(s.get("history", [])) >= question_limit
