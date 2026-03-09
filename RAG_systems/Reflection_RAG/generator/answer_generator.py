"""
Answer Generator: produce an initial answer from retrieved context using OpenAI ChatCompletion.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a helpful assistant that answers questions based only on the provided context.
If the context does not contain enough information to answer fully, say so and answer only from what is given.
Do not invent facts or cite information not present in the context.
Be clear and concise."""


class AnswerGenerator:
    """Generate an initial answer given a question and retrieved context chunks. Works with OpenAI or Azure OpenAI."""

    def __init__(
        self,
        client,  # OpenAI or AzureOpenAI
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, question: str, context_chunks: List[str]) -> str:
        """
        Produce an initial answer using the question and retrieved context.
        """
        context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(No context provided)"
        user_content = f"Context:\n{context}\n\nQuestion: {question}\n\nProvide an answer based only on the context above."
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            answer = (resp.choices[0].message.content or "").strip()
            logger.info("Generated initial answer (%s chars)", len(answer))
            return answer
        except Exception as e:
            logger.exception("Answer generation failed: %s", e)
            raise RuntimeError("Failed to generate initial answer") from e
