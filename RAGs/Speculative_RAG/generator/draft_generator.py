"""
Draft generator: small/fast model produces an initial answer from the user question only.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_DRAFT_SYSTEM = (
    "You are a helpful assistant. Answer the user's question concisely and accurately. "
    "If you are unsure, say so. Do not invent facts."
)


def generate_draft(
    client,
    model: str,
    question: str,
    *,
    system_prompt: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> str:
    """
    Generate a fast draft answer using the small model.
    Uses only the user question (no retrieved context) for speed.

    Args:
        client: OpenAI or AzureOpenAI client.
        model: Draft model name or Azure deployment name.
        question: User question.
        system_prompt: Optional system message; uses default if None.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens to generate.

    Returns:
        Draft answer string.
    """
    messages = [
        {"role": "system", "content": system_prompt or DEFAULT_DRAFT_SYSTEM},
        {"role": "user", "content": question},
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        logger.info("Draft generated (%s chars)", len(text))
        return text
    except Exception as e:
        logger.exception("Draft generation failed: %s", e)
        raise
