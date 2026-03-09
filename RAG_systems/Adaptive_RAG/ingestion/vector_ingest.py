"""
Vector ingestion: load documents, create embeddings, store in FAISS.
"""

import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from config import (
    AZURE_BASE_URL,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT,
    DATA_DIR,
    EMBEDDING_DIM,
    FAISS_INDEX_PATH,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def load_documents(data_dir: Path | None = None) -> list[str]:
    """Load text from .txt and .md files in data_dir."""
    data_dir = Path(data_dir or DATA_DIR)
    if not data_dir.is_dir():
        logger.warning("Data dir not found: %s", data_dir)
        return []
    texts = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        suf = path.suffix.lower()
        if suf in (".txt", ".md"):
            try:
                texts.append(path.read_text(encoding="utf-8", errors="replace"))
            except Exception as e:
                logger.warning("Failed to read %s: %s", path, e)
    return texts


def chunk_text(documents: list[str], chunk_size: int = 500, overlap: int = 50) -> list[str]:
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
            start = end - overlap
            if start <= 0:
                start = end
    return chunks


def create_embeddings(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Create embeddings via Azure OpenAI."""
    from openai import AzureOpenAI
    if not AZURE_OPENAI_API_KEY or not AZURE_BASE_URL:
        raise ValueError("Azure credentials not set")
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_BASE_URL,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(input=batch, model=AZURE_OPENAI_EMBEDDING_DEPLOYMENT)
        out.extend([item.embedding for item in resp.data])
    return out


def build_faiss_index(chunks: list[str], embeddings: list[list[float]], index_path: Path | None = None) -> None:
    """Build FAISS index and save; save chunk texts alongside."""
    import numpy as np
    try:
        import faiss
    except ImportError as e:
        raise ImportError("faiss-cpu required: pip install faiss-cpu") from e
    if not chunks or len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length must match")
    index_path = Path(index_path or FAISS_INDEX_PATH)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    dim = len(embeddings[0])
    matrix = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(matrix)
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)
    faiss.write_index(index, str(index_path))
    docs_path = index_path.parent / "documents.txt"
    _SEP = "\n<<<CHUNK>>>\n"
    docs_path.write_text(_SEP.join(chunks), encoding="utf-8")
    logger.info("Wrote FAISS index to %s (%d vectors)", index_path, len(chunks))


def run(data_dir: Path | None = None) -> None:
    """Load docs, chunk, embed, build FAISS index."""
    docs = load_documents(data_dir)
    if not docs:
        logger.warning("No documents loaded")
        return
    chunks = chunk_text(docs)
    if not chunks:
        logger.warning("No chunks produced")
        return
    logger.info("Creating embeddings for %d chunks...", len(chunks))
    embeddings = create_embeddings(chunks)
    build_faiss_index(chunks, embeddings)
    logger.info("Vector ingestion complete.")


if __name__ == "__main__":
    run()
    sys.exit(0)
