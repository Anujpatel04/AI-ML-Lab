"""
Hybrid retriever combining vector and graph retrieval with deduplication and ranking.
"""

from .vector_retriever import retrieve as vector_retrieve
from .graph_retriever import retrieve as graph_retrieve
from utils.logger import get_logger

logger = get_logger(__name__)


def retrieve(query: str, top_k_vector: int = 5, graph_limit: int = 10) -> str:
    """
    Perform hybrid retrieval: vector + graph. Merges and deduplicates, returns single context string.
    """
    query = (query or "").strip()
    if not query:
        return ""
    parts = []
    vec_results = vector_retrieve(query, top_k=top_k_vector)
    if vec_results:
        texts = [r["content"] for r in vec_results]
        parts.append("Document context:\n" + "\n\n".join(texts))
    graph_ctx = graph_retrieve(query, limit=graph_limit)
    if graph_ctx:
        parts.append(graph_ctx)
    if not parts:
        return ""
    return "\n\n".join(parts)
