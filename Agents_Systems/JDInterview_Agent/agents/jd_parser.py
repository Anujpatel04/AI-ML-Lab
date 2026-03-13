"""
JD Parser agent: extract role, skills, experience level, topics from job description.
"""

import logging
import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config.settings import AZURE_OPENAI_BASE_URL, AZURE_OPENAI_API_KEY, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION
from chains.llm_factory import get_llm
from prompts.jd_prompt import JD_EXTRACT_PROMPT
from models.schemas import ParsedJD
from utils.scoring import extract_json_from_text

logger = logging.getLogger(__name__)


def _parse_jd_response(text: str) -> ParsedJD:
    """Parse LLM response into ParsedJD."""
    data = extract_json_from_text(text)
    if not data or not isinstance(data, dict):
        return ParsedJD()
    return ParsedJD(
        role=str(data.get("role", "")).strip(),
        skills=[str(x).strip() for x in data.get("skills", []) if x],
        experience_level=str(data.get("experience_level", "")).strip(),
        topics=[str(x).strip() for x in data.get("topics", []) if x],
        key_responsibilities=[str(x).strip() for x in data.get("key_responsibilities", []) if x],
    )


def parse_jd(job_description: str) -> ParsedJD:
    """
    Extract structured role information from a job description.
    Returns ParsedJD with role, skills, experience_level, topics.
    """
    job_description = (job_description or "").strip()
    if not job_description:
        return ParsedJD()
    try:
        llm = get_llm(temperature=0.2)
        chain = JD_EXTRACT_PROMPT | llm | StrOutputParser()
        out = chain.invoke({"job_description": job_description})
        return _parse_jd_response(out)
    except Exception as e:
        logger.exception("JD parse failed: %s", e)
        return ParsedJD()
