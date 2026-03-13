"""
Azure OpenAI LLM factory for LangChain. Loads config from repo root .env.
"""

import logging
from typing import Any

from config.settings import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_BASE_URL,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
)

logger = logging.getLogger(__name__)
_llm: Any = None


def get_llm(temperature: float = 0.3):
    """Return a LangChain Azure Chat OpenAI instance."""
    global _llm
    if _llm is not None:
        return _llm
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_BASE_URL:
        raise ValueError(
            "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT (or AZURE_KEY, AZURE_ENDPOINT) in repo root .env"
        )
    try:
        from langchain_openai import AzureChatOpenAI
        _llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_BASE_URL,
            api_key=AZURE_OPENAI_API_KEY,
            deployment_name=AZURE_OPENAI_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=temperature,
        )
    except ImportError:
        from langchain_community.chat_models import AzureChatOpenAI
        _llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_BASE_URL,
            api_key=AZURE_OPENAI_API_KEY,
            deployment_name=AZURE_OPENAI_DEPLOYMENT,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=temperature,
        )
    return _llm
