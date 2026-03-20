# GitHub Issue Automation Agent

> **Automates issue triage:** classification, labels, priority, summary, root-cause hypothesis, fix suggestions, actionable tasks, and recommended owner—via one API call.

## Problem

Engineering teams spend time manually labeling, prioritizing, and breaking down GitHub issues. This service uses an LLM (Azure OpenAI by default) to produce **structured, consistent JSON** for automation or CI.

## Quick Start

```bash
source /path/to/Anuj-AI-ML-Lab/.venv/bin/activate
cd MCP_tools/GitHubIssueAgent_MCP
pip install -r requirements.txt
PYTHONPATH=. uvicorn api.main:app --reload --host 0.0.0.0 --port 8010
```

Uses repo root `.env` (`AZURE_ENDPOINT`, `AZURE_KEY`, `API_VERSION`, …). Optional: `ISSUE_AGENT_LLM_PROVIDER` = `azure` | `openai` | `ollama`.

## API

`POST /analyze-issue` — body:

```json
{
  "title": "Login API returns 500",
  "description": "Users cannot log in when calling /api/login.",
  "comments": ["Happens on prod only"],
  "labels": ["login"],
  "metadata": {"area": "auth"}
}
```

Response matches the strict schema: `issue_type`, `labels`, `priority`, `summary`, `root_cause`, `suggested_fix`, `tasks`, `recommended_assignee`.

Open **http://localhost:8010/docs** to try it interactively.

## Project layout

| Path | Role |
|------|------|
| `api/main.py` | FastAPI app |
| `llm/chains.py` | LLM + JSON parse + validation |
| `llm/prompts.py` | System/human prompts |
| `models/schemas.py` | Request/response Pydantic models |
| `utils/config.py` | Env (Azure / OpenAI / Ollama) |
