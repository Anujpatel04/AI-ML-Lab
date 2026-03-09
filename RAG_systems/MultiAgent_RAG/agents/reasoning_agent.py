"""
Reasoning Agent: synthesizes a structured answer from retrieved context using an LLM.
"""

from openai import AzureOpenAI

from core.prompts import REASONING_SYSTEM, REASONING_USER_TEMPLATE
from utils.config import (
    AZURE_BASE_URL,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class ReasoningAgent:
    """Uses Azure OpenAI to generate an answer grounded in the provided context."""

    def __init__(self):
        self._azure_client: AzureOpenAI | None = None

    def _get_client(self) -> AzureOpenAI:
        if self._azure_client is None:
            if not AZURE_OPENAI_API_KEY or not AZURE_BASE_URL:
                raise ValueError("Azure OpenAI credentials not set in .env")
            self._azure_client = AzureOpenAI(
                api_key=AZURE_OPENAI_API_KEY,
                azure_endpoint=AZURE_BASE_URL,
                api_version=AZURE_OPENAI_API_VERSION,
            )
        return self._azure_client

    def run(self, query: str, context: str) -> str:
        """
        Receives retrieved context and user query; returns a structured answer.
        """
        query = (query or "").strip()
        context = (context or "").strip()
        if not query:
            return ""
        if not context:
            return "No relevant context was retrieved. Please add documents to the knowledge base or rephrase your question."
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": REASONING_SYSTEM},
                    {"role": "user", "content": REASONING_USER_TEMPLATE.format(context=context, query=query)},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            answer = (resp.choices[0].message.content or "").strip()
            logger.info("Reasoning agent produced answer (%d chars)", len(answer))
            return answer
        except Exception as e:
            logger.exception("Reasoning agent failed: %s", e)
            raise
