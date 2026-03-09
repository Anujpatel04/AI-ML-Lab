"""
Reflection Agent: critique the initial answer (hallucinations, missing context)
and produce an improved final answer using OpenAI.
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

REFLECTION_SYSTEM = """You are a senior reviewer. Your job is to:
1. Critique the given answer against the provided context.
2. Identify any hallucinations (claims not supported by the context), missing information, or unclear parts.
3. Then produce an improved answer that stays grounded in the context, fills gaps, and removes inaccuracies.

Output your response in this exact format:

CRITIQUE:
<your critique and list of issues found>

IMPROVED_ANSWER:
<the improved answer text>"""


class ReflectionAgent:
    """Critique initial answer and return improved answer grounded in context. Works with OpenAI or Azure OpenAI."""

    def __init__(
        self,
        client,  # OpenAI or AzureOpenAI
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def critique_and_improve(
        self,
        question: str,
        context_chunks: List[str],
        initial_answer: str,
    ) -> Tuple[str, str]:
        """
        Critique the initial answer and return (critique_text, improved_answer).
        """
        context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(No context)"
        user_content = (
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Initial answer to review:\n{initial_answer}\n\n"
            "Provide your CRITIQUE and then IMPROVED_ANSWER in the format specified."
        )
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": REFLECTION_SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            raw = (resp.choices[0].message.content or "").strip()
            critique, improved = self._parse_response(raw)
            logger.info("Reflection produced critique (%s chars) and improved answer (%s chars)", len(critique), len(improved))
            return critique, improved
        except Exception as e:
            logger.exception("Reflection failed: %s", e)
            raise RuntimeError("Reflection step failed") from e

    def _parse_response(self, raw: str) -> Tuple[str, str]:
        """Extract CRITIQUE and IMPROVED_ANSWER sections from model output."""
        import re
        critique = ""
        improved = ""
        critique_match = re.search(r"CRITIQUE:\s*(.*?)(?=IMPROVED_ANSWER:|\Z)", raw, re.DOTALL | re.IGNORECASE)
        if critique_match:
            critique = critique_match.group(1).strip()
        improved_match = re.search(r"IMPROVED_ANSWER:\s*(.*)", raw, re.DOTALL | re.IGNORECASE)
        if improved_match:
            improved = improved_match.group(1).strip()
        if not improved:
            improved = raw
        return critique or "(No critique extracted)", improved or "(No improved answer extracted)"
