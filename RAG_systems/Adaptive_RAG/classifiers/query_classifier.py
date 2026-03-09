"""
LLM-based query classifier for Adaptive RAG. Classifies into factual, entity, or broad.
"""

import json
import re
from typing import Literal

from openai import AzureOpenAI

from config import (
    AZURE_BASE_URL,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
)
from utils.logger import get_logger

logger = get_logger(__name__)

QueryType = Literal["factual", "entity", "broad"]

CLASSIFIER_SYSTEM = """You are a query classifier. Classify the user's question into exactly one of:
- factual: factual or semantic questions (what is X, how does Y work, define Z)
- entity: questions about entities and their relationships (who worked with X, what is the relationship between A and B)
- broad: complex or multi-facet questions that need both factual and relationship context

Respond with a JSON object only: {"query": "<original query>", "type": "factual"|"entity"|"broad"}"""


def _get_client() -> AzureOpenAI:
    if not AZURE_OPENAI_API_KEY or not AZURE_BASE_URL:
        raise ValueError("AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT (or AZURE_KEY, AZURE_ENDPOINT) must be set")
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_BASE_URL,
        api_version=AZURE_OPENAI_API_VERSION,
    )


def classify_query(query: str) -> dict[str, str]:
    """
    Classify the query into factual, entity, or broad.
    Returns {"query": str, "type": "factual"|"entity"|"broad"}.
    """
    query = (query or "").strip()
    if not query:
        return {"query": "", "type": "factual"}
    client = _get_client()
    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": CLASSIFIER_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("Classification failed, defaulting to factual: %s", e)
        return {"query": query, "type": "factual"}
    match = re.search(r"\{[^{}]*\}", text)
    if not match:
        return {"query": query, "type": "factual"}
    try:
        data = json.loads(match.group())
        q = data.get("query", query)
        t = (data.get("type") or "factual").lower()
        if t not in ("factual", "entity", "broad"):
            t = "factual"
        return {"query": q, "type": t}
    except (json.JSONDecodeError, TypeError):
        return {"query": query, "type": "factual"}
