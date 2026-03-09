"""
Verification Agent: validates that the answer is grounded in retrieved documents.
"""

import json
import re

from openai import AzureOpenAI

from core.prompts import (
    REFINEMENT_SYSTEM,
    REFINEMENT_USER_TEMPLATE,
    VERIFICATION_SYSTEM,
    VERIFICATION_USER_TEMPLATE,
)
from utils.config import (
    AZURE_BASE_URL,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
)
from utils.logger import get_logger

logger = get_logger(__name__)


class VerifierAgent:
    """Validates factual grounding and triggers refinement if verification fails."""

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

    def verify(self, context: str, answer: str) -> tuple[bool, str]:
        """
        Returns (verified: bool, reason: str).
        """
        context = (context or "").strip()
        answer = (answer or "").strip()
        if not answer:
            return True, "No answer to verify."
        if not context:
            return False, "No context to verify against."
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": VERIFICATION_SYSTEM},
                    {"role": "user", "content": VERIFICATION_USER_TEMPLATE.format(context=context, answer=answer)},
                ],
                temperature=0.0,
                max_tokens=256,
            )
            text = (resp.choices[0].message.content or "").strip()
            match = re.search(r"\{[^{}]*\}", text)
            if match:
                data = json.loads(match.group())
                verified = bool(data.get("verified", False))
                reason = str(data.get("reason", "")).strip() or ("Grounded" if verified else "Not grounded")
                return verified, reason
            return False, "Could not parse verification response."
        except Exception as e:
            logger.exception("Verification failed: %s", e)
            return False, str(e)

    def refine(self, context: str, answer: str, feedback: str) -> str:
        """Produce a revised answer that is grounded in the context."""
        context = (context or "").strip()
        answer = (answer or "").strip()
        feedback = (feedback or "").strip()
        if not context or not answer:
            return answer
        try:
            client = self._get_client()
            resp = client.chat.completions.create(
                model=AZURE_OPENAI_DEPLOYMENT,
                messages=[
                    {"role": "system", "content": REFINEMENT_SYSTEM},
                    {"role": "user", "content": REFINEMENT_USER_TEMPLATE.format(context=context, answer=answer, feedback=feedback)},
                ],
                temperature=0.2,
                max_tokens=1024,
            )
            return (resp.choices[0].message.content or answer).strip()
        except Exception as e:
            logger.warning("Refinement failed, returning original: %s", e)
            return answer
