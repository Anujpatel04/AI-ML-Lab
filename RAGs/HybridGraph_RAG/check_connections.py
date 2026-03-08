"""
Test Neo4j, Pinecone, and OpenAI/Azure connections with short timeouts.
Run from repo root: python -m RAGs.HybridGraph_RAG.check_connections
Or from this dir:  python check_connections.py
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Ensure repo root .env is loaded
def _bootstrap():
    from pathlib import Path
    app_dir = Path(__file__).resolve().parent
    repo_root = app_dir.parent.parent
    dotenv = repo_root / ".env"
    if dotenv.is_file():
        from dotenv import load_dotenv
        load_dotenv(dotenv)


_bootstrap()

TIMEOUT = 12  # seconds per check


def _get_config():
    try:
        from RAGs.HybridGraph_RAG import config
        return config
    except Exception:
        import config
        return config


def _test_neo4j() -> tuple[bool, str]:
    try:
        config = _get_config()
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            config.NEO4J_URI,
            auth=(config.NEO4J_USERNAME, config.NEO4J_PASSWORD),
            connection_timeout=5,
        )
        with driver.session() as session:
            session.run("RETURN 1 AS n").single()
        driver.close()
        return True, "OK"
    except Exception as e:
        return False, str(e)


def _test_pinecone() -> tuple[bool, str]:
    try:
        config = _get_config()
        if not config.PINECONE_API_KEY:
            return False, "PINECONE_API_KEY not set"
        from pinecone import Pinecone
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        indexes = pc.list_indexes()
        names = indexes.names() if hasattr(indexes, "names") else []
        if config.PINECONE_INDEX in names:
            idx = pc.Index(config.PINECONE_INDEX)
            idx.describe_index_stats()
        return True, "OK"
    except Exception as e:
        return False, str(e)


def _test_openai_azure() -> tuple[bool, str]:
    try:
        config = _get_config()
        client, chat_model, embedding_model = config.get_rag_client()
        # Short embedding
        r = client.embeddings.create(
            model=embedding_model,
            input=["test"],
        )
        if not r.data or not getattr(r.data[0], "embedding", None):
            return False, "Empty embedding response"
        # Short chat
        r2 = client.chat.completions.create(
            model=chat_model,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=10,
        )
        if not r2.choices:
            return False, "Empty chat response"
        return True, "OK"
    except Exception as e:
        return False, str(e)


def run_checks():
    print("Hybrid Graph RAG — connection checks (timeout per check: %ss)\n" % TIMEOUT)
    results = []

    # Neo4j
    print("Neo4j ... ", end="", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_test_neo4j)
            ok, msg = fut.result(timeout=TIMEOUT)
    except FuturesTimeoutError:
        ok, msg = False, "timeout"
    results.append(("Neo4j", ok, msg))
    print("OK" if ok else f"FAIL: {msg}")

    # Pinecone
    print("Pinecone ... ", end="", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_test_pinecone)
            ok, msg = fut.result(timeout=TIMEOUT)
    except FuturesTimeoutError:
        ok, msg = False, "timeout"
    results.append(("Pinecone", ok, msg))
    print("OK" if ok else f"FAIL: {msg}")

    # OpenAI / Azure
    print("OpenAI/Azure (embed + chat) ... ", end="", flush=True)
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_test_openai_azure)
            ok, msg = fut.result(timeout=TIMEOUT)
    except FuturesTimeoutError:
        ok, msg = False, "timeout"
    results.append(("OpenAI/Azure", ok, msg))
    print("OK" if ok else f"FAIL: {msg}")

    print()
    failed = [r for r in results if not r[1]]
    if failed:
        print("Failed:", [r[0] for r in failed])
        return 1
    print("All connections OK.")
    return 0


if __name__ == "__main__":
    sys.exit(run_checks())
