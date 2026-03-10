"""
Self-Healing RAG pipeline: retrieve, evaluate relevance, optional retry, generate answer.
"""

from typing import Any, Optional

from openai import AzureOpenAI

import config
import logging
from config import (
    AZURE_BASE_URL,
    AZURE_CHAT_DEPLOYMENT,
    AZURE_EMBEDDING_DEPLOYMENT,
    AZURE_KEY,
    API_VERSION,
    DEFAULT_TOP_K,
    PINECONE_API_KEY,
    PINECONE_HOST,
    PINECONE_INDEX,
    PINECONE_NAMESPACE,
)

logger = logging.getLogger(__name__)

_OPENAI_CLIENT: Optional[AzureOpenAI] = None


def _get_client() -> AzureOpenAI:
    global _OPENAI_CLIENT
    if _OPENAI_CLIENT is None:
        if not AZURE_KEY or not AZURE_BASE_URL:
            raise ValueError("Set AZURE_KEY and AZURE_ENDPOINT (or AZURE_OPENAI_*) in repo root .env")
        _OPENAI_CLIENT = AzureOpenAI(
            api_key=AZURE_KEY,
            azure_endpoint=AZURE_BASE_URL,
            api_version=API_VERSION,
        )
    return _OPENAI_CLIENT


def _embed(texts: list[str], embedding_deployment: Optional[str] = None) -> list[list[float]]:
    if not texts:
        return []
    client = _get_client()
    model = embedding_deployment or AZURE_EMBEDDING_DEPLOYMENT
    resp = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in resp.data]


def _get_pinecone_index():
    if not PINECONE_API_KEY:
        return None
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        if PINECONE_HOST:
            return pc.Index(host=PINECONE_HOST)
        return pc.Index(PINECONE_INDEX)
    except Exception as e:
        logger.warning("Pinecone index connection failed: %s", e)
        return None


def retrieve_documents(
    query: str,
    top_k: int | None = None,
    embedding_deployment: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Retrieve top-k documents from Pinecone using query embedding.
    Returns list of {"content": str, "score": float}.
    """
    top_k = top_k or DEFAULT_TOP_K
    query = (query or "").strip()
    if not query:
        return []
    index = _get_pinecone_index()
    if index is None:
        return []
    try:
        q_emb = _embed([query], embedding_deployment=embedding_deployment)
        if not q_emb:
            return []
        result = index.query(
            vector=q_emb[0],
            top_k=top_k,
            namespace=PINECONE_NAMESPACE,
            include_metadata=True,
        )
        out = []
        matches = getattr(result, "matches", None) or []
        for match in matches:
            meta = getattr(match, "metadata", None) or {}
            content = meta.get("text") if isinstance(meta, dict) else (getattr(match, "content", None) or "")
            score_val = getattr(match, "score", None)
            score = float(score_val) if score_val is not None else 0.0
            if content:
                out.append({"content": content, "score": score})
        return out
    except Exception as e:
        logger.exception("Pinecone retrieve_documents failed: %s", e)
        return []


def evaluate_relevance(
    query: str,
    documents: list[dict[str, Any]],
    threshold: float | None = None,
    chat_deployment: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Use LLM to evaluate whether retrieved documents are relevant enough.
    Returns (is_sufficient: bool, message: str).
    """
    threshold = threshold if threshold is not None else config.DEFAULT_RELEVANCE_THRESHOLD
    query = (query or "").strip()
    if not documents:
        return False, "No documents retrieved."
    context = "\n\n".join(d.get("content", "")[:800] for d in documents[:10])
    prompt = f"""Evaluate whether the following retrieved documents are relevant and sufficient to answer the user question. Consider relevance and coverage.

User question: {query}

Retrieved documents:
{context[:4000]}

Respond with exactly two lines:
LINE1: YES or NO (whether retrieval is sufficient)
LINE2: Brief reason (one sentence)"""
    try:
        client = _get_client()
        model = chat_deployment or AZURE_CHAT_DEPLOYMENT
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        text = (resp.choices[0].message.content or "").strip().upper()
        is_yes = "YES" in text.split("\n")[0] if text else False
        reason = "\n".join(text.split("\n")[1:]).strip() if "\n" in text else text
        return is_yes, reason or ("Sufficient" if is_yes else "Insufficient relevance")
    except Exception as e:
        return False, str(e)


def retry_retrieval(
    query: str,
    top_k: int | None = None,
    embedding_deployment: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Perform a new retrieval (e.g. with higher top_k). Used when initial retrieval is poor.
    """
    top_k = top_k or min(DEFAULT_TOP_K * 2, 20)
    return retrieve_documents(query, top_k=top_k, embedding_deployment=embedding_deployment)


def generate_answer(
    query: str,
    documents: list[dict[str, Any]],
    chat_deployment: Optional[str] = None,
) -> str:
    """Generate final answer from query and retrieved documents using Azure OpenAI."""
    query = (query or "").strip()
    if not query:
        return ""
    context = "\n\n".join(d.get("content", "") for d in documents) if documents else ""
    if not context:
        return "No relevant documents were retrieved to answer this question."
    prompt = f"""Use the following context to answer the user question. Base your answer only on the provided context. If the context does not contain enough information, say so.

Context:
{context[:12000]}

User question: {query}

Answer:"""
    try:
        client = _get_client()
        model = chat_deployment or AZURE_CHAT_DEPLOYMENT
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"Error generating answer: {e}"
