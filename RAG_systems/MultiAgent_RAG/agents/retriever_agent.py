"""
Retriever Agent: fetches top-k relevant documents using the vector store.
"""

from typing import Any

from core.vector_store import search
from utils.logger import get_logger

logger = get_logger(__name__)


class RetrieverAgent:
    """Uses embeddings and FAISS to retrieve relevant documents for a query."""

    def __init__(self, top_k: int | None = None):
        self.top_k = top_k

    def run(self, query: str) -> list[dict[str, Any]]:
        """
        Accepts the user query, runs similarity search, returns top-k documents.
        Each item: {"content": str, "score": float}.
        """
        query = (query or "").strip()
        if not query:
            return []
        results = search(query, top_k=self.top_k)
        logger.info("Retriever returned %d documents", len(results))
        return results
