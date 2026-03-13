"""
Follow-up Generator agent: generate deeper question based on candidate answer.
"""

import logging

from chains.llm_factory import get_llm
from prompts.followup_prompt import FOLLOWUP_PROMPT
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


def generate_followup(question: str, answer: str) -> str:
    """
    Generate a single follow-up question that probes deeper based on the candidate's answer.
    """
    question = (question or "").strip()
    answer = (answer or "").strip()
    if not question or not answer:
        return ""
    try:
        llm = get_llm(temperature=0.4)
        chain = FOLLOWUP_PROMPT | llm | StrOutputParser()
        out = chain.invoke({"question": question, "answer": answer})
        return (out or "").strip()[:500]
    except Exception as e:
        logger.exception("Follow-up generation failed: %s", e)
        return ""
