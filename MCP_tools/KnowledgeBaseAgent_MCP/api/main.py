"""
Knowledge Base Agent — RAG API over internal documents.
POST /ingest — index a folder; POST /query — ask a question.
"""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, HTTPException

from core.document_loader import load_documents_from_directory, split_documents
from core.rag_chain import answer_question
from core.vector_store import collection_count, ingest_documents
from models.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse, SourceRef
from utils.config import KB_DOCS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Knowledge Base Agent (RAG)",
    description="Answer questions from internal documents (markdown, text, PDF).",
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "indexed_chunks": collection_count()}


@app.get("/")
def root() -> dict:
    return {
        "service": "knowledge-base-rag",
        "docs": "/docs",
        "ingest": "POST /ingest",
        "query": "POST /query",
    }


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    """
    Load .md / .txt / .pdf from ``docs_path`` (default: kb_documents/), chunk, embed into Chroma.
    """
    base = Path(req.docs_path).resolve() if req.docs_path else KB_DOCS_DIR
    try:
        raw = load_documents_from_directory(base)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    chunks = split_documents(raw)
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail=f"No supported documents found under {base} (.md, .txt, .pdf).",
        )
    try:
        n = ingest_documents(chunks, reset=req.reset)
    except Exception as e:
        logger.exception("ingest_failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    return IngestResponse(status="success", chunks_indexed=n, docs_path=str(base))


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Answer a question using retrieved context from the knowledge base."""
    try:
        out = answer_question(req.question)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        logger.exception("query_failed")
        raise HTTPException(status_code=500, detail=str(e)) from e

    sources = [SourceRef(**s) for s in out["sources"]]
    return QueryResponse(answer=out["answer"], sources=sources)
