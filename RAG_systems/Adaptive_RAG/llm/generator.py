"""
LLM answer generator using Azure OpenAI chat completion.
"""

from openai import AzureOpenAI

from config import (
    AZURE_BASE_URL,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
)
from utils.logger import get_logger

logger = get_logger(__name__)

PROMPT_TEMPLATE = """You are a knowledge assistant.

Use the provided context to answer the user's question.

Context:
{context}

Question:
{query}

Answer concisely and accurately."""


def _get_client() -> AzureOpenAI:
    if not AZURE_OPENAI_API_KEY or not AZURE_BASE_URL:
        raise ValueError("Azure credentials not set")
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_BASE_URL,
        api_version=AZURE_OPENAI_API_VERSION,
    )


def generate(context: str, query: str) -> str:
    """
    Generate answer from context and query using Azure OpenAI.
    Returns the model response text.
    """
    query = (query or "").strip()
    context = (context or "").strip()
    if not query:
        return ""
    prompt = PROMPT_TEMPLATE.format(context=context or "(No context provided.)", query=query)
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=1024,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.exception("Generation failed: %s", e)
        raise
