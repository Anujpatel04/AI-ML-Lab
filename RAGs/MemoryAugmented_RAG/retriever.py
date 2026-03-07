"""
Handles document embedding and vector retrieval from Pinecone.
Documents are loaded from the data/ folder, chunked, embedded with OpenAI, and stored
in a dedicated Pinecone namespace. retrieve_documents() returns top-k relevant chunks.
"""

import hashlib
import logging
from pathlib import Path
from typing import List, Optional

from utils.document_loader import load_documents, chunk_text

logger = logging.getLogger(__name__)


def _get_embeddings(client, model: str, texts: List[str], batch_size: int = 100) -> List[List[float]]:
    """Generate embeddings in batches."""
    out = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(input=batch, model=model)
        for item in resp.data:
            out.append(item.embedding)
    return out


def index_documents(
    index,
    client,
    embedding_model: str,
    data_dir: Path,
    *,
    namespace: str = "documents",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    id_prefix: str = "doc",
) -> int:
    """
    Load documents from data_dir, chunk, embed, and upsert into Pinecone.
    Call this once (or when documents change) to build the knowledge base.

    Args:
        index: Pinecone index object.
        client: OpenAI (or compatible) client for embeddings.
        embedding_model: Embedding model name.
        data_dir: Directory containing .txt, .md (and optionally .pdf) files.
        namespace: Pinecone namespace for documents.
        chunk_size: Character size per chunk.
        chunk_overlap: Overlap between consecutive chunks.
        id_prefix: Prefix for vector IDs.

    Returns:
        Number of chunks upserted. Returns 0 if no documents found.
    """
    data_dir = Path(data_dir)
    docs = load_documents(data_dir)
    if not docs:
        logger.warning("No documents found in %s", data_dir)
        return 0
    chunks = chunk_text(docs, chunk_size=chunk_size, overlap=chunk_overlap)
    if not chunks:
        return 0
    embeddings = _get_embeddings(client, embedding_model, chunks)
    vectors = []
    for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
        # Stable ID from content so re-indexing overwrites instead of duplicating
        h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:16]
        vec_id = f"{id_prefix}_{i}_{h}"
        vectors.append({"id": vec_id, "values": vec, "metadata": {"text": chunk}})
    # Upsert in batches (Pinecone accepts batches)
    batch_size = 100
    for j in range(0, len(vectors), batch_size):
        batch = vectors[j : j + batch_size]
        index.upsert(vectors=batch, namespace=namespace)
    logger.info("Indexed %s document chunks to namespace %s", len(vectors), namespace)
    return len(vectors)


def retrieve_documents(
    index,
    client,
    embedding_model: str,
    query: str,
    *,
    namespace: str = "documents",
    top_k: int = 5,
) -> List[str]:
    """
    Retrieve top-k relevant document chunks from Pinecone for a given query.

    Args:
        index: Pinecone index object.
        client: OpenAI (or compatible) client for embeddings.
        embedding_model: Embedding model name.
        query: User query string.
        namespace: Pinecone namespace for documents.
        top_k: Maximum number of chunks to return.

    Returns:
        List of chunk text strings, most relevant first. Empty list if none or on error.
    """
    if not (query or "").strip():
        return []
    query = query.strip()
    embeddings = _get_embeddings(client, embedding_model, [query])
    if not embeddings:
        return []
    try:
        result = index.query(
            namespace=namespace,
            vector=embeddings[0],
            top_k=top_k,
            include_metadata=True,
        )
        texts = []
        for m in getattr(result, "matches", []):
            meta = getattr(m, "metadata", None) or {}
            if isinstance(meta, dict) and "text" in meta:
                texts.append(meta["text"])
            elif hasattr(m, "metadata") and hasattr(m.metadata, "get"):
                t = m.metadata.get("text")
                if t:
                    texts.append(t)
        return texts
    except Exception as e:
        logger.warning("retrieve_documents failed: %s", e)
        return []
