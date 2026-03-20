"""
GitHub Issue Automation Agent — FastAPI service.
POST /analyze-issue: classify, summarize, suggest fixes, tasks, priority, assignee.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from llm.chains import analyze_issue
from models.schemas import IssueAnalysisResult, IssueInput
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="GitHub Issue Automation Agent",
    description="Analyze GitHub issues: labels, priority, summary, root cause, tasks, assignee.",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/")
def root() -> dict:
    return {"service": "github-issue-automation", "docs": "/docs", "analyze": "POST /analyze-issue"}


@app.post("/analyze-issue", response_model=IssueAnalysisResult)
def analyze_issue_endpoint(issue: IssueInput) -> IssueAnalysisResult:
    """
    Analyze a GitHub issue and return structured JSON (issue_type, labels, priority, summary, ...).
    """
    try:
        return analyze_issue(issue)
    except ValueError as e:
        logger.exception("analyze_issue | parse_error")
        raise HTTPException(status_code=422, detail=f"Invalid LLM output: {e}") from e
    except Exception as e:
        logger.exception("analyze_issue | error")
        raise HTTPException(status_code=500, detail=str(e)) from e
