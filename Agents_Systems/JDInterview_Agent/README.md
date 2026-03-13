# Adaptive AI Interview Simulator

Production-quality AI interview simulator: role-adaptive technical interviews from a job description. Uses **LangChain**, **FastAPI**, and **Azure OpenAI**.

## Features

- **JD Parser:** Extract role, skills, experience level, topics from job description
- **Question Generator:** Role-specific technical questions (configurable difficulty)
- **Answer Evaluator:** Score (1–10), strengths, weaknesses, feedback
- **Follow-up Generator:** Deeper questions from candidate answers
- **Final Report:** Overall score, strengths, gaps, recommended topics

## Setup

- **Environment:** Repo root `.env` with Azure OpenAI:
  - `AZURE_OPENAI_API_KEY` or `AZURE_KEY`
  - `AZURE_OPENAI_ENDPOINT` or `AZURE_ENDPOINT`
  - `AZURE_OPENAI_API_VERSION` or `API_VERSION`
- **Optional:** `INTERVIEW_QUESTION_LIMIT` (default 5)

## Run

**Streamlit frontend (recommended):**
```bash
cd Agents_Systems/JDInterview_Agent
pip install -r requirements.txt
streamlit run frontend.py
```
Or: `./run_frontend.sh` (uses repo `.venv` if present). Open http://localhost:8501

**FastAPI backend only:**
```bash
uvicorn main:app --reload
```
API docs: http://localhost:8000/docs

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | /api/parse-jd | Parse job description → role, skills, topics |
| POST | /api/generate-questions | Generate questions from role + skills + difficulty |
| POST | /api/evaluate-answer | Evaluate answer → score, feedback |
| POST | /api/follow-up | Generate follow-up question |
| POST | /api/final-report | Generate final report from session history |

## Project Structure

- `config/settings.py` – Azure config (loads repo `.env`)
- `agents/` – JD parser, question generator, evaluator, follow-up, final report
- `chains/` – LLM factory, interview orchestration
- `prompts/` – Prompt templates
- `api/interview_routes.py` – FastAPI routes
- `memory/session_memory.py` – In-memory session store (Redis-ready)
