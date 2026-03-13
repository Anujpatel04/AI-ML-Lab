"""
Question Generator agent: generate role-specific technical interview questions.
"""

import logging
import re

from chains.llm_factory import get_llm
from prompts.question_prompt import QUESTION_GEN_PROMPT
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)


def generate_questions(role: str, skills: list[str], difficulty: str = "medium") -> list[str]:
    """
    Generate technical interview questions for the given role and skills.
    Returns a list of 5 questions.
    """
    role = (role or "").strip()
    skills_str = ", ".join(skills) if skills else "general technical skills"
    difficulty = (difficulty or "medium").strip().lower()
    if difficulty not in ("easy", "medium", "hard"):
        difficulty = "medium"
    try:
        llm = get_llm(temperature=0.5)
        chain = QUESTION_GEN_PROMPT | llm | StrOutputParser()
        out = chain.invoke({
            "role": role or "Technical role",
            "skills": skills_str,
            "difficulty": difficulty,
        })
        lines = [s.strip() for s in (out or "").strip().split("\n") if s.strip()]
        questions = []
        for line in lines:
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
            if cleaned and len(cleaned) > 10:
                questions.append(cleaned)
        return questions[:5]
    except Exception as e:
        logger.exception("Question generation failed: %s", e)
        return []
