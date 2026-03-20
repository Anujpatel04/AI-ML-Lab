# GitHub Issue Automation Agent

FastAPI service that turns raw issue text into **structured triage data** (labels, priority, summary, tasks, owner hints) using an LLM.

## Problem

Issue backlogs are hard to keep consistent: humans must guess type, severity, labels, and next steps—often slowly and unevenly across teams.

## How it works

1. **Input** — You `POST` issue fields (`title`, `description`, optional `comments` / `labels` / `metadata`).
2. **Analysis** — LangChain calls your configured model (defaults to **Azure OpenAI** from the repo root `.env`).
3. **Output** — Validated JSON: `issue_type`, `labels`, `priority`, `summary`, `root_cause`, `suggested_fix`, `tasks` (3–6 steps), `recommended_assignee`.

Use the response for automation, Slack/GitHub bots, or CI—not as a substitute for human review on critical issues.

## Quick start

```bash
source /path/to/Anuj-AI-ML-Lab/.venv/bin/activate
cd MCP_tools/GitHubIssueAgent_MCP
pip install -r requirements.txt
PYTHONPATH=. uvicorn api.main:app --reload --host 0.0.0.0 --port 8010
# or: ./run.sh
```

Config: repo root `.env` (`AZURE_ENDPOINT`, `AZURE_KEY`, `API_VERSION`, …). Optional: `ISSUE_AGENT_LLM_PROVIDER` = `azure` | `openai` | `ollama`.

## API

**`POST /analyze-issue`** — body example:

```json
{
  "title": "Login API returns 500",
  "description": "Users cannot log in when calling POST /api/login.",
  "comments": ["Reproduces on prod only"],
  "labels": ["auth"],
  "metadata": {"env": "production"}
}
```

## Layout

| Path | Role |
|------|------|
| `api/main.py` | HTTP routes |
| `llm/chains.py` | LLM call, JSON extract, validation |
| `llm/prompts.py` | Prompts |
| `models/schemas.py` | Request / response schemas |
| `utils/config.py` | Environment |
