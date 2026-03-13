"""
Final Report Generator: produce interview evaluation report from session history.
"""

import json
import logging
from typing import Any

from chains.llm_factory import get_llm
from prompts.final_report_prompt import FINAL_REPORT_PROMPT
from langchain_core.output_parsers import StrOutputParser
from models.schemas import FinalReport
from utils.scoring import extract_json_from_text

logger = logging.getLogger(__name__)


def generate_final_report(session_history: list[dict[str, Any]]) -> FinalReport:
    """
    Generate final interview evaluation report from session history.
    session_history: list of {"question", "answer", "score", "feedback"} dicts.
    """
    if not session_history:
        return FinalReport(detailed_feedback="No session history provided.")
    try:
        history_str = json.dumps(session_history, indent=2)
        llm = get_llm(temperature=0.3)
        chain = FINAL_REPORT_PROMPT | llm | StrOutputParser()
        out = chain.invoke({"session_history": history_str})
        data = extract_json_from_text(out)
        if not data or not isinstance(data, dict):
            return FinalReport(detailed_feedback=out or "Could not parse report.")
        return FinalReport(
            role=str(data.get("role", "")).strip(),
            overall_score=str(data.get("overall_score", "")).strip(),
            technical_strengths=[str(x).strip() for x in data.get("technical_strengths", []) if x],
            knowledge_gaps=[str(x).strip() for x in data.get("knowledge_gaps", []) if x],
            recommended_topics=[str(x).strip() for x in data.get("recommended_topics", []) if x],
            detailed_feedback=str(data.get("detailed_feedback", "")).strip(),
        )
    except Exception as e:
        logger.exception("Final report generation failed: %s", e)
        return FinalReport(detailed_feedback=str(e))
