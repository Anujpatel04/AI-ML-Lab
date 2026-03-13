"""
Prompt for Answer Evaluator: score and feedback for candidate answers.
"""

from langchain_core.prompts import PromptTemplate

EVALUATION_PROMPT = PromptTemplate.from_template("""You are an expert technical interviewer evaluating a candidate's answer.

Question: {question}
Candidate's answer: {answer}

Evaluate the answer and return a valid JSON object only (no markdown, no code block) with these exact keys:
- "score": integer from 1 to 10 (10 = excellent, 1 = poor)
- "strengths": array of strings, what the candidate did well
- "weaknesses": array of strings, gaps or errors
- "feedback": string, brief improvement suggestions

Output only the JSON object, nothing else.""")
