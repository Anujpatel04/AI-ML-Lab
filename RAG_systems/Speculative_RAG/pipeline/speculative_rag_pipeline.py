"""
Speculative RAG pipeline: retrieval -> draft generation -> verification -> final answer.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from retriever.retriever import Retriever
from generator.draft_generator import generate_draft
from verifier.verifier_agent import verify_and_improve

logger = logging.getLogger(__name__)


@dataclass
class SpeculativeRAGResult:
    """Result of a single speculative RAG run."""
    question: str
    context_chunks: List[str]
    draft_answer: str
    verification_feedback: str
    final_answer: str


class SpeculativeRAGPipeline:
    """
    Orchestrates: retrieve top-k chunks -> draft (small model) -> verify (large model) -> final answer.
    """

    def __init__(
        self,
        retriever: Retriever,
        client,
        draft_model: str,
        verifier_model: str,
        top_k: int = 5,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        self.retriever = retriever
        self.client = client
        self.draft_model = draft_model
        self.verifier_model = verifier_model
        self.top_k = top_k
        self.temperature = temperature
        self.max_tokens = max_tokens

    def run(self, question: str) -> SpeculativeRAGResult:
        """
        Execute full pipeline for one question.

        Returns:
            SpeculativeRAGResult with context_chunks, draft_answer, verification_feedback, final_answer.
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("Question cannot be empty.")

        # 1. Retrieve relevant chunks
        try:
            context_chunks = self.retriever.retrieve(question, top_k=self.top_k)
        except Exception as e:
            logger.exception("Retrieval failed: %s", e)
            raise RuntimeError(f"Retrieval failed: {e}") from e

        # 2. Draft answer (small model, question only)
        try:
            draft_answer = generate_draft(
                self.client,
                self.draft_model,
                question,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.exception("Draft generation failed: %s", e)
            raise RuntimeError(f"Draft generation failed: {e}") from e

        # 3. Verify and improve (large model, with context)
        try:
            final_answer, feedback = verify_and_improve(
                self.client,
                self.verifier_model,
                question,
                context_chunks,
                draft_answer,
                temperature=0.2,
                max_tokens=self.max_tokens,
            )
        except Exception as e:
            logger.exception("Verification failed: %s", e)
            final_answer = draft_answer
            feedback = f"Verification step failed; using draft. Error: {e}"

        return SpeculativeRAGResult(
            question=question,
            context_chunks=context_chunks,
            draft_answer=draft_answer,
            verification_feedback=feedback,
            final_answer=final_answer,
        )
