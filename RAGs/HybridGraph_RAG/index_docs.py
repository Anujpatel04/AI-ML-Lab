#!/usr/bin/env python3
"""
Index data/ into Pinecone from the command line. Use this if "Index documents now" in the app times out.
Use the repo .venv. From repo root (unbuffered output to see where it stops if killed):
  .venv/bin/python -u RAGs/HybridGraph_RAG/index_docs.py
If the process is killed (OOM), try closing other apps or run "Index documents now" in the Streamlit app instead.
"""
import sys
import os

# Unbuffered output so we see where the process stops if killed
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(line_buffering=True)


def _check_connections():
    """Verify OpenAI/Azure and Pinecone before indexing. Returns (True, None) or (False, error_msg)."""
    try:
        import config
        client, _, embedding_model = config.get_rag_client()
        # Quick embeddings test (one token)
        resp = client.embeddings.create(model=embedding_model, input=["test"], timeout=15)
        if not resp.data or not getattr(resp.data[0], "embedding", None):
            return False, "Embeddings API returned empty response"
    except Exception as e:
        return False, f"OpenAI/Azure connection failed: {e}"

    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        indexes = pc.list_indexes()
        names = indexes.names() if hasattr(indexes, "names") else []
        if config.PINECONE_INDEX not in names:
            # Index might be created by run_index_documents
            pass
    except Exception as e:
        return False, f"Pinecone connection failed: {e}"

    return True, None


def main():
    try:
        import pinecone  # noqa: F401
    except ImportError:
        print("pinecone not installed. Use the repo .venv:", file=sys.stderr)
        print("  cd /path/to/Anuj-AI-ML-Lab && .venv/bin/python RAGs/HybridGraph_RAG/index_docs.py", file=sys.stderr)
        return 1

    print("Checking connections (OpenAI/Azure + Pinecone)...", flush=True)
    ok, err = _check_connections()
    if not ok:
        print(f"Connection check failed: {err}", file=sys.stderr)
        print("Fix .env (OPENAI_API_KEY or AZURE_* / PINECONE_API_KEY) and try again.", file=sys.stderr)
        return 1
    print("Connections OK.", flush=True)

    def progress(msg):
        print(msg, flush=True)

    try:
        from pathlib import Path
        # Data dir: same folder as this script -> data/
        _script_dir = Path(__file__).resolve().parent
        data_dir = (_script_dir / "data").resolve()
        if not data_dir.is_dir():
            print(f"Data directory not found: {data_dir}", file=sys.stderr)
            return 1
        progress(f"Data directory: {data_dir}")

        from rag_pipeline import config as _config
        from vector_store import index_documents
        progress("Loading client & Pinecone index...")
        client, _, embedding_model = _config.get_rag_client()
        from rag_pipeline import _get_pinecone_index
        index = _get_pinecone_index(skip_ensure=False)
        progress("Indexing documents from data/...")
        n = index_documents(
            index, client, embedding_model, data_dir,
            namespace=_config.PINECONE_NAMESPACE_DOCS,
            chunk_size=max(_config.CHUNK_SIZE, 1000),
            chunk_overlap=min(_config.CHUNK_OVERLAP, 120),
            progress_callback=progress,
            embedding_timeout=_config.OPENAI_TIMEOUT,
            batch_size=5,
            use_parallel_embeddings=False,
        )
        print(f"Done. Indexed {n} chunks. You can now use the app to ask questions.", flush=True)
        return 0
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
