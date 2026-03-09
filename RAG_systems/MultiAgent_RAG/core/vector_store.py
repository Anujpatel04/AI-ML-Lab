"""
Vector store using SentenceTransformers and FAISS for similarity search.
"""

import re
from pathlib import Path
from typing import Any

from utils.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    DOCUMENTS_STORE_PATH,
    EMBEDDING_MODEL,
    FAISS_INDEX_PATH,
    TOP_K,
)
from utils.logger import get_logger

logger = get_logger(__name__)

_CHUNK_SEP = "\n<<<CHUNK>>>\n"


def _data_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "data"


def _load_documents(data_dir: Path | None = None) -> list[str]:
    """Load text from .txt and .md files."""
    data_dir = Path(data_dir or _data_dir() / "documents")
    if not data_dir.is_dir():
        return []
    texts = []
    for p in sorted(data_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in (".txt", ".md"):
            try:
                texts.append(p.read_text(encoding="utf-8", errors="replace"))
            except Exception as e:
                logger.warning("Failed to read %s: %s", p, e)
    return texts


def _chunk_text(documents: list[str], chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Split documents into overlapping chunks."""
    if overlap >= chunk_size:
        overlap = chunk_size // 2
    chunks = []
    for doc in documents:
        doc = (doc or "").replace("\r\n", "\n").strip()
        if not doc:
            continue
        start = 0
        while start < len(doc):
            end = min(start + chunk_size, len(doc))
            if end < len(doc):
                br = doc.rfind("\n", start, end + 1)
                if br > start:
                    end = br + 1
                else:
                    br = doc.rfind(" ", start, end + 1)
                    if br > start:
                        end = br + 1
            block = doc[start:end].strip()
            if block:
                chunks.append(block)
            if end >= len(doc):
                break
            start = max(end - overlap, start + 1)
    return chunks


def _get_embeddings(texts: list[str], model_name: str) -> list[list[float]]:
    """Embed texts using SentenceTransformers."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError("sentence-transformers required: pip install sentence-transformers") from e
    model = SentenceTransformer(model_name)
    return model.encode(texts, convert_to_numpy=True).tolist()


def build_index(data_dir: Path | None = None) -> int:
    """Load docs, chunk, embed, build FAISS index. Returns number of chunks indexed."""
    try:
        import faiss
        import numpy as np
    except ImportError as e:
        raise ImportError("faiss-cpu required: pip install faiss-cpu") from e
    data_dir = data_dir or _data_dir() / "documents"
    docs = _load_documents(data_dir)
    if not docs:
        logger.warning("No documents in %s", data_dir)
        return 0
    chunks = _chunk_text(docs, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        return 0
    logger.info("Embedding %d chunks with %s", len(chunks), EMBEDDING_MODEL)
    vectors = _get_embeddings(chunks, EMBEDDING_MODEL)
    index_path = Path(FAISS_INDEX_PATH)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.array(vectors, dtype=np.float32)
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    faiss.write_index(index, str(index_path))
    store_path = index_path.parent / "documents.txt"
    store_path.write_text(_CHUNK_SEP.join(chunks), encoding="utf-8")
    logger.info("Built FAISS index with %d vectors at %s", len(chunks), index_path)
    return len(chunks)


def search(query: str, top_k: int | None = None) -> list[dict[str, Any]]:
    """
    Run similarity search. Returns list of {"content": str, "score": float}.
    Builds index from data/documents if index does not exist.
    """
    top_k = top_k or TOP_K
    query = (query or "").strip()
    if not query:
        return []
    data_dir = _data_dir()
    index_path = data_dir / "faiss_index"
    docs_path = data_dir / "documents.txt"
    if not index_path.exists() or not docs_path.exists():
        n = build_index(data_dir / "documents")
        if n == 0:
            return []
    try:
        import faiss
        import numpy as np
    except ImportError:
        logger.warning("faiss-cpu not installed")
        return []
    index = faiss.read_index(str(index_path))
    raw = docs_path.read_text(encoding="utf-8", errors="replace")
    docs = [s.strip() for s in raw.split(_CHUNK_SEP) if s.strip()]
    if not docs:
        return []
    vectors = _get_embeddings([query], EMBEDDING_MODEL)
    q_vec = np.array([vectors[0]], dtype=np.float32)
    faiss.normalize_L2(q_vec)
    k = min(top_k, index.ntotal)
    scores, indices = index.search(q_vec, k)
    results = []
    for i, idx in enumerate(indices[0]):
        if 0 <= idx < len(docs) and docs[idx]:
            results.append({"content": docs[idx], "score": float(scores[0][i])})
    return results
