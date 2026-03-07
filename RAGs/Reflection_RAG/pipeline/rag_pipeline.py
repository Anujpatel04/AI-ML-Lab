"""
RAG Pipeline: orchestrate retrieval -> initial answer -> reflection (critique + improved answer).
Uses OpenAI or Azure OpenAI from config (repo root .env).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import config
from retriever import Retriever
from generator import AnswerGenerator
from reflection import ReflectionAgent

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Single entry point: given a question, return retrieved context,
    initial answer, reflection critique, and final improved answer.
    """

    def __init__(
        self,
        data_dir: Path,
        index_path: Path,
        chunks_path: Path,
        top_k: int = 5,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ):
        client, chat_model, embedding_model = config.get_rag_client()
        self.retriever = Retriever(
            data_dir=data_dir,
            index_path=index_path,
            chunks_path=chunks_path,
            client=client,
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        self.generator = AnswerGenerator(
            client=client,
            model=chat_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self.reflection = ReflectionAgent(
            client=client,
            model=chat_model,
            temperature=0.2,
            max_tokens=max_tokens,
        )
        self.top_k = top_k

    def run(self, question: str) -> Dict[str, Any]:
        """
        Execute: retrieve -> generate -> reflect -> return.
        Keys: context_chunks, initial_answer, critique, final_answer.
        """
        question = (question or "").strip()
        if not question:
            return {
                "context_chunks": [],
                "initial_answer": "",
                "critique": "",
                "final_answer": "",
                "error": "Question is empty.",
            }

        try:
            context_chunks: List[str] = self.retriever.retrieve(question, top_k=self.top_k)
        except Exception as e:
            logger.exception("Retrieval failed: %s", e)
            return {
                "context_chunks": [],
                "initial_answer": "",
                "critique": "",
                "final_answer": "",
                "error": str(e),
            }

        try:
            initial_answer = self.generator.generate(question, context_chunks)
        except Exception as e:
            logger.exception("Generation failed: %s", e)
            return {
                "context_chunks": context_chunks,
                "initial_answer": "",
                "critique": "",
                "final_answer": "",
                "error": str(e),
            }

        try:
            critique, final_answer = self.reflection.critique_and_improve(
                question, context_chunks, initial_answer
            )
        except Exception as e:
            logger.exception("Reflection failed: %s", e)
            return {
                "context_chunks": context_chunks,
                "initial_answer": initial_answer,
                "critique": "",
                "final_answer": initial_answer,
                "error": str(e),
            }

        return {
            "context_chunks": context_chunks,
            "initial_answer": initial_answer,
            "critique": critique,
            "final_answer": final_answer,
            "error": None,
        }
