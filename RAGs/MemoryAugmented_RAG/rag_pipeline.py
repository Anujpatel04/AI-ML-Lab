"""
Orchestrates the full Memory-Augmented RAG pipeline:
  memory retrieval -> document retrieval -> LLM answer generation -> store new memory.
"""

import logging
from typing import Optional

import config
from memory_store import store_memory, search_memory
from retriever import index_documents, retrieve_documents

logger = logging.getLogger(__name__)

# System prompt for the LLM: use memories and documents to answer.
RAG_SYSTEM_PROMPT = """You are a helpful assistant with access to relevant past conversations and document context.
Use the provided "Relevant past conversations" and "Relevant documents" to answer the user's question accurately.
If the context does not contain enough information, say so. Do not invent facts. Be concise."""


def _get_openai_client():
    """Return OpenAI client using API key from config (loaded from repo root .env)."""
    if not config.OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in the repo root .env file.")
    from openai import OpenAI
    return OpenAI(api_key=config.OPENAI_API_KEY)


def _get_pinecone_index():
    """Build Pinecone client and return the index. Uses PINECONE_INDEX; optional PINECONE_HOST for legacy."""
    if not config.PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY is not set in the repo root .env file.")
    from pinecone import Pinecone
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    if config.PINECONE_HOST:
        index = pc.Index(host=config.PINECONE_HOST)
    else:
        index = pc.Index(config.PINECONE_INDEX)
    return index


def _generate_answer(client, query: str, memory_texts: list, doc_texts: list) -> str:
    """Build prompt from memories + documents and call the LLM."""
    memories_block = "\n".join(memory_texts) if memory_texts else "(None)"
    docs_block = "\n\n".join(doc_texts) if doc_texts else "(None)"
    user_content = f"""Relevant past conversations:
{memories_block}

Relevant documents:
{docs_block}

Question: {query}

Answer the question using the above context when relevant."""

    resp = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )
    return (resp.choices[0].message.content or "").strip()


def run(
    query: str,
    *,
    index_docs_if_empty: bool = True,
    top_k_memory: Optional[int] = None,
    top_k_docs: Optional[int] = None,
) -> str:
    """
    Run the full Memory-Augmented RAG pipeline for one user query.

    1. Retrieve relevant memories from Pinecone (namespace "memory").
    2. Retrieve relevant documents from Pinecone (namespace "documents").
    3. Generate a final response with the LLM using memories + documents.
    4. Store the interaction (Q + A) as new memory in Pinecone.

    Args:
        query: User question.
        index_docs_if_empty: If True, call index_documents once when documents namespace is empty (e.g. first run).
        top_k_memory: Override config for number of memories to retrieve.
        top_k_docs: Override config for number of document chunks to retrieve.

    Returns:
        The generated answer string.
    """
    query = (query or "").strip()
    if not query:
        raise ValueError("Query cannot be empty.")

    top_k_memory = top_k_memory if top_k_memory is not None else config.TOP_K_MEMORY
    top_k_docs = top_k_docs if top_k_docs is not None else config.TOP_K_DOCS

    client = _get_openai_client()
    index = _get_pinecone_index()

    # Optional: ensure document index is populated from data/ on first use
    if index_docs_if_empty and config.DATA_DIR.is_dir():
        try:
            stats = index.describe_index_stats()
            namespaces = getattr(stats, "namespaces", None) or {}
            ns_stats = namespaces.get(config.NAMESPACE_DOCUMENTS)
            vector_count = getattr(ns_stats, "vector_count", 0) if ns_stats else 0
            if (vector_count or 0) == 0:
                index_documents(
                    index,
                    client,
                    config.EMBEDDING_MODEL,
                    config.DATA_DIR,
                    namespace=config.NAMESPACE_DOCUMENTS,
                    chunk_size=config.CHUNK_SIZE,
                    chunk_overlap=config.CHUNK_OVERLAP,
                )
        except Exception as e:
            logger.warning("Could not check or build document index: %s", e)

    # 1. Memory search
    memory_texts = search_memory(
        index,
        client,
        config.EMBEDDING_MODEL,
        query,
        namespace=config.NAMESPACE_MEMORY,
        top_k=top_k_memory,
    )

    # 2. Document retrieval
    doc_texts = retrieve_documents(
        index,
        client,
        config.EMBEDDING_MODEL,
        query,
        namespace=config.NAMESPACE_DOCUMENTS,
        top_k=top_k_docs,
    )

    # 3. LLM answer generation
    answer = _generate_answer(client, query, memory_texts, doc_texts)

    # 4. Store new memory (conversation turn)
    memory_text = f"Q: {query}\nA: {answer}"
    try:
        store_memory(
            index,
            client,
            config.EMBEDDING_MODEL,
            memory_text,
            namespace=config.NAMESPACE_MEMORY,
        )
    except Exception as e:
        logger.warning("Failed to store memory: %s", e)

    return answer


def main():
    """CLI entry: read query from argv or stdin and print the answer."""
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = sys.stdin.read().strip()
    if not q:
        print("Usage: python rag_pipeline.py 'Your question'", file=sys.stderr)
        sys.exit(1)
    try:
        answer = run(q)
        print(answer)
    except Exception as e:
        logger.exception("Pipeline failed")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
