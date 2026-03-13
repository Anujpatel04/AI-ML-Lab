"""
Answer Evaluator agent: score and provide feedback on candidate answers.
"""

import logging
from typing import Any

from chains.llm_factory import get_llm
from prompts.evaluation_prompt import EVALUATION_PROMPT
from langchain_core.output_parsers import StrOutputParser
from models.schemas import EvaluationResult
from utils.scoring import extract_json_from_text, normalize_score

logger = logging.getLogger(__name__)


def evaluate_answer(question: str, answer: str) -> EvaluationResult:
    """
    Evaluate a candidate answer. Returns score (1-10), strengths, weaknesses, feedback.
    """
    question = (question or "").strip()
    answer = (answer or "").strip()
    if not question or not answer:
        return EvaluationResult(score=5, feedback="No question or answer provided.")
    try:
        llm = get_llm(temperature=0.2)
        chain = EVALUATION_PROMPT | llm | StrOutputParser()
        out = chain.invoke({"question": question, "answer": answer})
        data = extract_json_from_text(out)
        if not data or not isinstance(data, dict):
            return EvaluationResult(score=5, feedback=out or "Could not parse evaluation.")
        score = normalize_score(data.get("score"), 1, 10)
        strengths = [str(x).strip() for x in data.get("strengths", []) if x]
        weaknesses = [str(x).strip() for x in data.get("weaknesses", []) if x]
        feedback = str(data.get("feedback", "")).strip()
        return EvaluationResult(
            score=score,
            strengths=strengths,
            weaknesses=weaknesses,
            feedback=feedback,
        )
    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        return EvaluationResult(score=5, feedback=str(e))
