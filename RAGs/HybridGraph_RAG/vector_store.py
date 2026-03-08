"""
Handle embeddings and vector retrieval in Pinecone.
Supports indexing documents from data/ and semantic search for the hybrid RAG.
"""

import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

from utils.document_loader import load_documents, chunk_text

logger = logging.getLogger(__name__)

# Parallel embedding requests (2–3 at a time) to speed up indexing
EMBEDDING_MAX_WORKERS = 3


def _embed_one_batch(client, model: str, batch: List[str], timeout: float, batch_idx: int):
    """Call embeddings API for one batch; returns (batch_idx, list of embeddings)."""
    kwargs = {"input": batch, "model": model}
    if timeout > 0:
        try:
            resp = client.embeddings.create(**kwargs, timeout=timeout)
        except TypeError:
            resp = client.embeddings.create(**kwargs)
    else:
        resp = client.embeddings.create(**kwargs)
    return (batch_idx, [item.embedding for item in resp.data])


def _get_embeddings(
    client,
    model: str,
    texts: List[str],
    batch_size: int = 100,
    timeout: float = 60.0,
    progress_callback=None,
    max_workers: int = EMBEDDING_MAX_WORKERS,
) -> List[List[float]]:
    """Generate embeddings via OpenAI/Azure API. Uses parallel requests when multiple batches."""
    if not texts:
        return []
    batches = [texts[i : i + batch_size] for i in range(0, len(texts), batch_size)]
    total_batches = len(batches)
    if total_batches == 1:
        # Single batch: no need for threads
        try:
            _, embs = _embed_one_batch(client, model, batches[0], timeout, 0)
            return embs
        except Exception as e:
            logger.warning("Embeddings API call failed: %s", e)
            raise
    # Multiple batches: run in parallel (respect rate limits with max_workers)
    workers = min(max_workers, total_batches)
    out = [None] * total_batches
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_embed_one_batch, client, model, batch, timeout, i): i
            for i, batch in enumerate(batches)
        }
        for fut in as_completed(futures):
            batch_idx, embs = fut.result()
            out[batch_idx] = embs
            done += 1
            if progress_callback:
                try:
                    progress_callback(f"Embeddings batch {done}/{total_batches}...")
                except Exception:
                    pass
    return [e for batch_embs in out for e in batch_embs]


def index_documents(
    index,
    client,
    embedding_model: str,
    data_dir: Path,
    *,
    namespace: str = "documents",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    progress_callback=None,
    embedding_timeout: float = 60.0,
    batch_size: int = 50,
    use_parallel_embeddings: bool = True,
) -> int:
    """
    Load docs from data_dir, chunk, then for each batch: embed and upsert (streaming to limit memory).
    Returns number of chunks indexed. progress_callback(msg) for UI updates.
    """
    def _progress(msg: str):
        if progress_callback:
            try:
                progress_callback(msg)
            except Exception:
                pass

    data_dir = Path(data_dir).resolve()
    _progress(f"Loading documents from: {data_dir}")
    if not data_dir.is_dir():
        _progress(f"Data directory does not exist: {data_dir}")
        logger.warning("Data dir not found: %s", data_dir)
        return 0
    docs = load_documents(data_dir)
    if not docs:
        _progress("No .txt/.md files found in data/. Add files and try again.")
        logger.warning("No documents in %s", data_dir)
        return 0
    total_chars = sum(len(d) for d in docs)
    _progress(f"Found {len(docs)} document(s) ({total_chars} chars), chunking...")
    chunks = chunk_text(docs, chunk_size=chunk_size, overlap=chunk_overlap)
    if not chunks:
        _progress("No text chunks produced (docs may be empty or chunk_size too large).")
        logger.warning("chunk_text returned 0 chunks for %s docs (%s chars)", len(docs), total_chars)
        return 0
    _progress(f"Got {len(chunks)} chunks. Starting embedding...")

    total_indexed = 0
    num_batches = (len(chunks) + batch_size - 1) // batch_size
    max_workers = EMBEDDING_MAX_WORKERS if use_parallel_embeddings else 1

    for b in range(0, len(chunks), batch_size):
        batch_chunks = chunks[b : b + batch_size]
        batch_num = (b // batch_size) + 1
        _progress(f"Embedding batch {batch_num}/{num_batches} ({len(batch_chunks)} chunks)...")
        try:
            embeddings = _get_embeddings(
                client, embedding_model, batch_chunks,
                batch_size=min(100, len(batch_chunks)),
                timeout=embedding_timeout,
                progress_callback=None,
                max_workers=max_workers if len(batch_chunks) > 100 else 1,
            )
        except Exception as e:
            logger.exception("Embeddings API failed")
            raise RuntimeError(f"Embeddings API failed: {e}") from e

        vectors = []
        for i, (chunk, vec) in enumerate(zip(batch_chunks, embeddings)):
            j = b + i
            h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()[:16]
            vec_id = f"doc_{j}_{h}"
            vectors.append({"id": vec_id, "values": vec, "metadata": {"text": chunk}})

        _progress(f"Upserting batch {batch_num}/{num_batches} to Pinecone...")
        try:
            index.upsert(vectors=vectors, namespace=namespace)
        except Exception as e:
            logger.exception("Pinecone upsert failed")
            raise RuntimeError(f"Pinecone upsert failed: {e}") from e

        total_indexed += len(vectors)

    logger.info("Indexed %s chunks to Pinecone namespace %s", total_indexed, namespace)
    return total_indexed


def retrieve_documents(
    index,
    client,
    embedding_model: str,
    query: str,
    *,
    namespace: str = "documents",
    top_k: int = 5,
) -> List[str]:
    """
    Semantic search in Pinecone. Returns top-k document chunk texts.
    """
    if not (query or "").strip():
        return []
    query = query.strip()
    emb = _get_embeddings(client, embedding_model, [query])
    if not emb:
        return []
    try:
        result = index.query(namespace=namespace, vector=emb[0], top_k=top_k, include_metadata=True)
        texts = []
        for m in getattr(result, "matches", []):
            meta = getattr(m, "metadata", None) or {}
            if isinstance(meta, dict) and "text" in meta:
                texts.append(meta["text"])
            elif hasattr(m, "metadata") and hasattr(m.metadata, "get"):
                t = m.metadata.get("text")
                if t:
                    texts.append(t)
        return texts
    except Exception as e:
        logger.warning("Vector retrieval failed: %s", e)
        return []
