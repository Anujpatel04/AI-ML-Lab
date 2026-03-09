"""
Orchestrator: coordinates retriever -> reasoning -> verification pipeline.
"""

from dataclasses import dataclass, field

from agents import RetrieverAgent, ReasoningAgent, VerifierAgent
from utils.config import TOP_K
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineResult:
    """Structured output from the multi-agent pipeline."""

    query: str = ""
    retrieved_docs: list[dict] = field(default_factory=list)
    context: str = ""
    reasoning_output: str = ""
    verified: bool = False
    verification_reason: str = ""
    final_answer: str = ""
    refined: bool = False


class Orchestrator:
    """Runs the full pipeline and maintains structured response objects."""

    def __init__(self, top_k: int | None = None):
        self.retriever = RetrieverAgent(top_k=top_k or TOP_K)
        self.reasoning = ReasoningAgent()
        self.verifier = VerifierAgent()

    def run(self, query: str) -> PipelineResult:
        """
        Execute: retriever -> reasoning -> verification (with optional refinement).
        """
        result = PipelineResult(query=(query or "").strip())
        if not result.query:
            result.final_answer = "Please provide a question."
            return result

        # 1. Retriever
        result.retrieved_docs = self.retriever.run(result.query)
        result.context = "\n\n".join(d["content"] for d in result.retrieved_docs if d.get("content"))
        if not result.context:
            result.final_answer = "No relevant documents were found. Add documents to data/documents/ and rebuild the index, or rephrase your question."
            return result

        # 2. Reasoning
        result.reasoning_output = self.reasoning.run(result.query, result.context)
        result.final_answer = result.reasoning_output

        # 3. Verification
        result.verified, result.verification_reason = self.verifier.verify(
            result.context, result.final_answer
        )
        if not result.verified:
            result.final_answer = self.verifier.refine(
                result.context, result.final_answer, result.verification_reason
            )
            result.refined = True
            result.verified, result.verification_reason = self.verifier.verify(
                result.context, result.final_answer
            )

        return result
