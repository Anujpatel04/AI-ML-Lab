"""
Orchestrate the full Hybrid Graph RAG pipeline:
  entity extraction → graph search → vector retrieval → context fusion → LLM answer.
"""

import logging
import time
from typing import Optional

import config
from hybrid_retriever import hybrid_retrieve
from vector_store import index_documents, retrieve_documents

logger = logging.getLogger(__name__)

RAG_SYSTEM = """You are a helpful assistant. Use the following context to answer the user's question.
Context may include knowledge graph (entities and relationships) and relevant document passages.
If the context does not contain enough information, say so. Do not invent facts. Be concise."""


def _ensure_pinecone_index(pc, index_name: str, dimension: int):
    """Create Pinecone serverless index if it does not exist."""
    try:
        existing = pc.list_indexes()
        names = existing.names() if hasattr(existing, "names") else []
        if index_name in names:
            return
    except Exception:
        names = []
    if index_name not in names:
        from pinecone import ServerlessSpec, Metric
        cloud = config.PINECONE_CLOUD
        region = config.PINECONE_REGION or ("us-east-1" if cloud == "aws" else "us-central1")
        spec = ServerlessSpec(cloud=cloud, region=region)
        pc.create_index(name=index_name, dimension=dimension, metric=Metric.COSINE, spec=spec)
        logger.info("Created Pinecone index %s. Waiting for ready (max 30s)...", index_name)
        for _ in range(6):
            time.sleep(5)
            try:
                d = pc.describe_index(index_name)
                if getattr(d, "status", None) and getattr(d.status, "ready", False):
                    break
            except Exception:
                pass


def _get_pinecone_index(*, skip_ensure: bool = False):
    """Return Pinecone index; create if missing unless skip_ensure=True (e.g. in Streamlit to avoid long create/wait)."""
    if not config.PINECONE_API_KEY:
        raise ValueError("PINECONE_API_KEY not set in repo root .env")
    from pinecone import Pinecone
    pc = Pinecone(api_key=config.PINECONE_API_KEY)
    if not skip_ensure:
        _ensure_pinecone_index(pc, config.PINECONE_INDEX, config.EMBEDDING_DIMENSION)
    return pc.Index(config.PINECONE_INDEX)


def run(
    query: str,
    *,
    index_docs_if_empty: bool = True,
    skip_pinecone_ensure: bool = False,
    progress_callback=None,
) -> str:
    """
    Run the full pipeline: extract entities → graph + vector retrieval → fuse → LLM answer.

    Args:
        query: User question.
        index_docs_if_empty: If True, index data/ into Pinecone when documents namespace is empty.
        skip_pinecone_ensure: If True, do not create Pinecone index if missing (use when index already exists, e.g. from Streamlit).
        progress_callback: Optional callable(message: str) for UI progress (e.g. "Step 1: ...").

    Returns:
        Generated answer string.
    """
    def _progress(msg: str):
        if progress_callback:
            try:
                progress_callback(msg)
            except Exception:
                pass

    query = (query or "").strip()
    if not query:
        raise ValueError("Query cannot be empty.")

    _progress("Loading client & Pinecone index...")
    client, chat_model, embedding_model = config.get_rag_client()
    index = _get_pinecone_index(skip_ensure=skip_pinecone_ensure)

    if index_docs_if_empty and config.DATA_DIR.is_dir():
        try:
            _progress("Checking document index...")
            stats = index.describe_index_stats()
            ns = getattr(stats, "namespaces", None) or {}
            ns_info = ns.get(config.PINECONE_NAMESPACE_DOCS)
            count = getattr(ns_info, "vector_count", 0) if ns_info else 0
            if (count or 0) == 0:
                _progress("Indexing documents (first run)...")
                try:
                    index_documents(
                        index,
                        client,
                        embedding_model,
                        config.DATA_DIR,
                        namespace=config.PINECONE_NAMESPACE_DOCS,
                        chunk_size=config.CHUNK_SIZE,
                        chunk_overlap=config.CHUNK_OVERLAP,
                        progress_callback=progress_callback,
                        embedding_timeout=60.0,
                    )
                except Exception as idx_err:
                    logger.warning("Could not index documents: %s", idx_err)
                    _progress("Indexing failed; continuing without docs.")
                    # Continue pipeline with empty doc context instead of failing
        except Exception as e:
            logger.warning("Could not check/build document index: %s", e)

    context = hybrid_retrieve(
        query,
        client=client,
        chat_model=chat_model,
        embedding_model=embedding_model,
        neo4j_uri=config.NEO4J_URI,
        neo4j_username=config.NEO4J_USERNAME,
        neo4j_password=config.NEO4J_PASSWORD,
        pinecone_index=index,
        vector_namespace=config.PINECONE_NAMESPACE_DOCS,
        progress_callback=progress_callback,
    )

    user_content = f"""Context:\n{context}\n\nQuestion: {query}\n\nAnswer using the context above."""

    _progress("Generating answer...")
    resp = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": RAG_SYSTEM},
            {"role": "user", "content": user_content},
        ],
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )
    return (resp.choices[0].message.content or "").strip()


def run_index_documents(progress_callback=None) -> int:
    """
    Load data/ and index into Pinecone (creates index if missing). Use once before querying.
    Returns number of chunks indexed.
    """
    def _progress(msg: str):
        if progress_callback:
            try:
                progress_callback(msg)
            except Exception:
                pass
    _progress("Loading client & Pinecone index...")
    client, _, embedding_model = config.get_rag_client()
    index = _get_pinecone_index(skip_ensure=False)
    data_dir = config.DATA_DIR.resolve()
    if not data_dir.is_dir():
        _progress("No data/ directory.")
        return 0
    _progress("Indexing documents from data/...")
    # Slightly larger chunks for indexing = fewer API calls, faster
    chunk_size = max(config.CHUNK_SIZE, 1000)
    chunk_overlap = min(config.CHUNK_OVERLAP, 120)
    return index_documents(
        index,
        client,
        embedding_model,
        data_dir,
        namespace=config.PINECONE_NAMESPACE_DOCS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        progress_callback=progress_callback,
        embedding_timeout=config.OPENAI_TIMEOUT,
    )


def main():
    """CLI: python rag_pipeline.py 'Your question'"""
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    q = " ".join(sys.argv[1:]).strip() if len(sys.argv) > 1 else sys.stdin.read().strip()
    if not q:
        print("Usage: python rag_pipeline.py 'Your question'", file=sys.stderr)
        sys.exit(1)
    try:
        print(run(q))
    except Exception as e:
        logger.exception("Pipeline failed")
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
