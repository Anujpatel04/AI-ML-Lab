"""
Interview orchestration: run question -> answer -> evaluate -> follow-up loop.
Lazy imports to avoid circular dependency with agents.
"""

import logging
from typing import Any

from models.schemas import ParsedJD, EvaluationResult, FinalReport

logger = logging.getLogger(__name__)


def run_interview_round(
    question: str,
    answer: str,
) -> tuple[EvaluationResult, str]:
    """
    Evaluate the answer and generate a follow-up question.
    Returns (evaluation_result, follow_up_question).
    """
    from agents import evaluate_answer, generate_followup
    eval_result = evaluate_answer(question, answer)
    followup = generate_followup(question, answer)
    return eval_result, followup


def get_questions_for_session(parsed: ParsedJD, difficulty: str = "medium") -> list[str]:
    """Generate question list for the session from parsed JD."""
    from agents import generate_questions
    return generate_questions(
        role=parsed.role,
        skills=parsed.skills,
        difficulty=difficulty,
    )


def build_final_report(session_history: list[dict[str, Any]]) -> FinalReport:
    """Build final report from session history."""
    from agents import generate_final_report
    return generate_final_report(session_history)
