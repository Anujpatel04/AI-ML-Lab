"""
Vector retriever using FAISS and Azure OpenAI embeddings.
"""

from pathlib import Path
from typing import Any

import numpy as np
from openai import AzureOpenAI

from config import (
    AZURE_BASE_URL,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    FAISS_INDEX_PATH,
    TOP_K_VECTOR,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def _get_client() -> AzureOpenAI:
    if not AZURE_OPENAI_API_KEY or not AZURE_BASE_URL:
        raise ValueError("Azure credentials not set")
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_BASE_URL,
        api_version=AZURE_OPENAI_API_VERSION,
    )


def _data_dir() -> Path:
    """Resolve data directory from this file's location so paths work from any cwd."""
    return Path(__file__).resolve().parent.parent / "data"


def _load_faiss_and_docs():
    """Load FAISS index and document store from data dir."""
    try:
        import faiss
    except ImportError:
        logger.warning("faiss-cpu not installed; install with: pip install faiss-cpu")
        return None, []
    data_dir = _data_dir()
    index_path = data_dir / "faiss_index"
    docs_path = data_dir / "documents.txt"
    if not index_path.exists():
        logger.warning("FAISS index not found at %s; run vector_ingest first", index_path)
        return None, []
    index = faiss.read_index(str(index_path))
    docs = []
    if docs_path.exists():
        _SEP = "\n<<<CHUNK>>>\n"
        raw = docs_path.read_text(encoding="utf-8", errors="replace")
        docs = raw.split(_SEP)
        # Trim whitespace but keep same count so indices match FAISS
        docs = [s.strip() for s in docs]
    if not docs or len(docs) < index.ntotal:
        logger.warning(
            "Document count (%d) < FAISS index size (%d); re-run vector_ingest",
            len(docs),
            index.ntotal,
        )
    return index, docs


def _embed(client: AzureOpenAI, texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    resp = client.embeddings.create(input=texts, model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
    return [item.embedding for item in resp.data]


def retrieve(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """
    Perform vector retrieval. Returns list of {"content": str, "score": float}.
    """
    top_k = top_k or TOP_K_VECTOR
    query = (query or "").strip()
    if not query:
        return []
    index, docs = _load_faiss_and_docs()
    if index is None or not docs:
        return []
    client = _get_client()
    q_emb = _embed(client, [query])
    if not q_emb:
        return []
    q_vec = np.array([q_emb[0]], dtype=np.float32)
    try:
        import faiss
        faiss.normalize_L2(q_vec)
    except Exception:
        pass
    k = min(top_k, index.ntotal)
    scores, indices = index.search(q_vec, k)
    results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(docs) and docs[idx]:
            score = float(scores[0][i]) if scores.size else 0.0
            results.append({"content": docs[idx], "score": score})
    return results
