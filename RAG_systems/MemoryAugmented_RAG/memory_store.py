"""
Handles storing and retrieving conversational memory from Pinecone.
Uses a dedicated namespace so memory is separate from the document index.
"""

import logging
import uuid
from typing import List, Optional

logger = logging.getLogger(__name__)


def _get_embeddings(client, model: str, texts: list[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts using the given OpenAI client and model."""
    if not texts:
        return []
    resp = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in resp.data]


def store_memory(
    index,
    client,
    embedding_model: str,
    text: str,
    *,
    namespace: str = "memory",
    id_prefix: str = "mem",
) -> str:
    """
    Store a single memory (e.g. "Q: ...\\nA: ...") in Pinecone.

    Args:
        index: Pinecone index object (from Pinecone(api_key=...).Index(index_name)).
        client: OpenAI (or compatible) client for embeddings.
        embedding_model: Embedding model name.
        text: Memory text to store (conversation turn or summary).
        namespace: Pinecone namespace for memories.
        id_prefix: Prefix for the generated vector ID.

    Returns:
        The assigned vector ID.
    """
    if not (text or "").strip():
        raise ValueError("Memory text cannot be empty.")
    text = text.strip()
    vectors = _get_embeddings(client, embedding_model, [text])
    if not vectors:
        raise RuntimeError("Failed to generate embedding for memory.")
    vec_id = f"{id_prefix}_{uuid.uuid4().hex}"
    index.upsert(
        vectors=[{"id": vec_id, "values": vectors[0], "metadata": {"text": text}}],
        namespace=namespace,
    )
    logger.info("Stored memory id=%s", vec_id)
    return vec_id


def search_memory(
    index,
    client,
    embedding_model: str,
    query: str,
    *,
    namespace: str = "memory",
    top_k: int = 5,
    include_metadata: bool = True,
) -> List[str]:
    """
    Retrieve relevant past memories from Pinecone for a given query.

    Args:
        index: Pinecone index object.
        client: OpenAI (or compatible) client for embeddings.
        embedding_model: Embedding model name.
        query: Search query (e.g. current user question).
        namespace: Pinecone namespace for memories.
        top_k: Maximum number of memories to return.
        include_metadata: If True, return full metadata (not used when returning list of strings).

    Returns:
        List of memory text strings, most relevant first. Returns empty list if none found or on error.
    """
    if not (query or "").strip():
        return []
    query = query.strip()
    vectors = _get_embeddings(client, embedding_model, [query])
    if not vectors:
        return []
    try:
        result = index.query(
            namespace=namespace,
            vector=vectors[0],
            top_k=top_k,
            include_metadata=include_metadata,
        )
        # Pinecone returns matches with .matches; each has .metadata with "text"
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
        logger.warning("search_memory failed: %s", e)
        return []
