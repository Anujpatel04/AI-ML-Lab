"""
FastAPI routes for Adaptive AI Interview Simulator.
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException

from agents import parse_jd, generate_questions, evaluate_answer, generate_followup, generate_final_report
from chains import get_questions_for_session, run_interview_round, build_final_report
from config.settings import DEFAULT_QUESTION_LIMIT
from memory import create_session, get_session, set_questions, append_turn, get_history, get_current_question, session_finished
from models.schemas import (
    ParseJDRequest,
    ParsedJD,
    GenerateQuestionsRequest,
    GenerateQuestionsResponse,
    EvaluateAnswerRequest,
    EvaluationResult,
    FollowUpRequest,
    FollowUpResponse,
    FinalReportRequest,
    FinalReport,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["interview"])


@router.post("/parse-jd", response_model=dict[str, Any])
def parse_job_description(body: ParseJDRequest) -> dict[str, Any]:
    """Parse job description and return structured role and skills."""
    try:
        parsed = parse_jd(body.job_description)
        return parsed.model_dump()
    except Exception as e:
        logger.exception("parse-jd failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate-questions", response_model=GenerateQuestionsResponse)
def generate_questions_endpoint(body: GenerateQuestionsRequest) -> GenerateQuestionsResponse:
    """Generate interview questions for role and skills."""
    try:
        questions = generate_questions(body.role, body.skills, body.difficulty)
        return GenerateQuestionsResponse(questions=questions)
    except Exception as e:
        logger.exception("generate-questions failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/evaluate-answer", response_model=dict[str, Any])
def evaluate_answer_endpoint(body: EvaluateAnswerRequest) -> dict[str, Any]:
    """Evaluate a candidate answer and return score and feedback."""
    try:
        result = evaluate_answer(body.question, body.answer)
        return {
            "score": result.score,
            "strengths": result.strengths,
            "weaknesses": result.weaknesses,
            "feedback": result.feedback,
        }
    except Exception as e:
        logger.exception("evaluate-answer failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/follow-up", response_model=FollowUpResponse)
def followup_endpoint(body: FollowUpRequest) -> FollowUpResponse:
    """Generate a follow-up question based on question and answer."""
    try:
        followup = generate_followup(body.question, body.answer)
        return FollowUpResponse(follow_up_question=followup)
    except Exception as e:
        logger.exception("follow-up failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/final-report", response_model=dict[str, Any])
def final_report_endpoint(body: FinalReportRequest) -> dict[str, Any]:
    """Generate final interview report from session history."""
    try:
        report = build_final_report(body.session_history)
        return report.model_dump()
    except Exception as e:
        logger.exception("final-report failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
