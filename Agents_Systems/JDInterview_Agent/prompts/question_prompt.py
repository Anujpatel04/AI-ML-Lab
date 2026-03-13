"""
Prompt for Question Generator: role-specific technical interview questions.
"""

from langchain_core.prompts import PromptTemplate

QUESTION_GEN_PROMPT = PromptTemplate.from_template("""You are an expert technical interviewer. Generate exactly 5 high-quality interview questions for the following role and context.

Role: {role}
Skills to cover: {skills}
Difficulty level: {difficulty}

Requirements:
- Questions must be technical and role-specific.
- Mix conceptual and practical questions.
- Align with the given skills and difficulty.
- Output exactly 5 questions, one per line, numbered 1. to 5.
- Do not include any other text or explanation, only the 5 questions.""")
