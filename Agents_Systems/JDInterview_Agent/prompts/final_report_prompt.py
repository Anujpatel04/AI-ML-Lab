"""
Prompt for Final Interview Report.
"""

from langchain_core.prompts import PromptTemplate

FINAL_REPORT_PROMPT = PromptTemplate.from_template("""You are an expert interview analyst. Given the following interview session history, produce a final evaluation report.

Session history (questions and answers with scores/feedback):
{session_history}

Return a valid JSON object only (no markdown, no code block) with these exact keys:
- "role": string, the interviewed role
- "overall_score": string, e.g. "7/10" or "Strong"
- "technical_strengths": array of strings
- "knowledge_gaps": array of strings
- "recommended_topics": array of strings for the candidate to improve
- "detailed_feedback": string, 2-3 paragraph summary

Output only the JSON object, nothing else.""")
