"""
Router: classify query, route to appropriate retriever, aggregate context, generate answer.
"""

from classifiers.query_classifier import classify_query
from retrievers import vector_retrieve, graph_retrieve, hybrid_retrieve
from llm.generator import generate
from utils.logger import get_logger

logger = get_logger(__name__)


def _aggregate_context(query: str, query_type: str) -> str:
    """Route to retriever and return aggregated context string."""
    if query_type == "factual":
        results = vector_retrieve(query)
        if not results:
            return ""
        return "\n\n".join(r["content"] for r in results)
    if query_type == "entity":
        ctx = graph_retrieve(query)
        if not ctx:
            results = vector_retrieve(query)
            if results:
                return "[Graph unavailable; using vector retrieval.]\n\n" + "\n\n".join(r["content"] for r in results)
        return ctx
    if query_type == "broad":
        ctx = hybrid_retrieve(query)
        if not ctx:
            results = vector_retrieve(query)
            if results:
                return "[Graph unavailable; using vector retrieval.]\n\n" + "\n\n".join(r["content"] for r in results)
        return ctx
    return vector_retrieve(query) or ""


def route(query: str) -> str:
    """
    Full pipeline: classify -> retrieve -> generate.
    Returns final answer string.
    """
    out = route_with_details(query)
    return out["answer"]


def route_with_details(query: str) -> dict:
    """
    Full pipeline with details for UI. Returns dict with keys:
    type (str), context (str), answer (str).
    """
    query = (query or "").strip()
    if not query:
        return {"type": "", "context": "", "answer": "Please provide a question."}
    classified = classify_query(query)
    query_type = classified.get("type", "factual")
    q = classified.get("query", query)
    logger.info("Query classified as %s", query_type)
    context = _aggregate_context(q, query_type)
    if not context:
        context = "(No relevant context found. You may need to run ingestion.)"
    answer = generate(context, q)
    return {"type": query_type, "context": context, "answer": answer}
