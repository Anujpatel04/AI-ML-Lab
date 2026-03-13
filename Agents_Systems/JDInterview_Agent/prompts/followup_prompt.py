"""
Prompt for Follow-up Generator: deeper question based on candidate answer.
"""

from langchain_core.prompts import PromptTemplate

FOLLOWUP_PROMPT = PromptTemplate.from_template("""You are an expert technical interviewer. Based on the candidate's answer, generate a single follow-up question that probes deeper into the topic.

Original question: {question}
Candidate's answer: {answer}

Generate exactly one follow-up question that:
- Tests conceptual depth or practical application
- Is relevant to what the candidate said
- Is concise and clear

Output only the follow-up question, nothing else. No numbering or prefix.""")
