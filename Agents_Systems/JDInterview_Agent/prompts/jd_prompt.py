"""
Prompt for JD Parser: extract structured role information from job description.
"""

from langchain_core.prompts import PromptTemplate

JD_EXTRACT_PROMPT = PromptTemplate.from_template("""You are an expert HR analyst. Extract structured information from the following job description.

Job description:
{job_description}

Extract and return a valid JSON object only (no markdown, no code block) with these exact keys:
- "role": string, the job title or role name
- "skills": array of strings, required technical and soft skills
- "experience_level": string, e.g. entry, mid, senior
- "topics": array of strings, main technical or domain topics for the role
- "key_responsibilities": array of strings, main responsibilities

Output only the JSON object, nothing else.""")
