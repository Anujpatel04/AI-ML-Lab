"""
FastAPI service for the AI Documentation Generator.
POST /generate-docs: read repo, parse, generate docs, save README, return path.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Add project root to path so imports work when running from api/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.doc_generator import generate_documentation
from utils.config import OUTPUT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="AI Documentation Generator",
    description="Generate professional documentation from a code repository using an LLM.",
    version="1.0.0",
)


class GenerateDocsRequest(BaseModel):
    """Request body for POST /generate-docs."""

    repo_path: str = Field(..., description="Path to the repository to document (e.g. ./sample_project)")


class GenerateDocsResponse(BaseModel):
    """Response for POST /generate-docs."""

    status: str = Field(..., description="'success' or 'error'")
    readme_path: str | None = Field(None, description="Path to the generated README.md")
    message: str | None = Field(None, description="Error or info message")


@app.post("/generate-docs", response_model=GenerateDocsResponse)
def generate_docs(request: GenerateDocsRequest) -> GenerateDocsResponse:
    """
    Generate documentation for the given repository.
    Workflow: read repo → parse code → generate docs with LLM → build markdown → save README.
    """
    repo_path = request.repo_path.strip()
    if not repo_path:
        raise HTTPException(status_code=400, detail="repo_path is required")

    resolved = Path(repo_path).resolve()
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Repository path does not exist: {repo_path}")
    if not resolved.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a directory: {repo_path}")

    try:
        readme_path = generate_documentation(str(resolved))
        return GenerateDocsResponse(
            status="success",
            readme_path=readme_path,
        )
    except FileNotFoundError as e:
        logger.exception("generate_docs | file_not_found")
        raise HTTPException(status_code=404, detail=str(e))
    except NotADirectoryError as e:
        logger.exception("generate_docs | not_a_directory")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("generate_docs | error")
        return GenerateDocsResponse(
            status="error",
            readme_path=None,
            message=str(e),
        )


@app.get("/")
def root() -> dict:
    """Root: API info and link to interactive docs."""
    return {
        "service": "AI Documentation Generator",
        "docs": "/docs",
        "health": "/health",
        "generate_docs": "POST /generate-docs",
    }


@app.get("/health")
def health() -> dict:
    """Health check."""
    return {"status": "ok"}
