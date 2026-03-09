"""
Hybrid Graph RAG — Streamlit frontend. Uses repo root .venv and .env.
Run: streamlit run app.py  (or use run_app.sh from repo root)
"""
import os
import sys


def _ensure_venv():
    """Re-exec with repo root .venv if pinecone is not available."""
    try:
        import pinecone  # noqa: F401
        return
    except ImportError:
        pass
    app_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(app_dir, "..", ".."))
    venv_python = os.path.join(repo_root, ".venv", "bin", "python")
    if os.path.isfile(venv_python):
        app_path = os.path.abspath(__file__)
        argv = [venv_python, "-m", "streamlit", "run", app_path]
        for a in sys.argv[1:]:
            if a != __file__ and (a != "app.py" or app_path not in argv):
                argv.append(a)
        os.execv(venv_python, argv)
    raise ImportError("pinecone not found. Use repo root .venv: source .venv/bin/activate then streamlit run app.py")


_ensure_venv()

import streamlit as st
from rag_pipeline import run

try:
    from rag_pipeline import run_index_documents
except ImportError:
    def run_index_documents(progress_callback=None):
        """Index data/ into Pinecone (fallback if not in rag_pipeline). Creates index if missing."""
        from pathlib import Path
        import config
        from pinecone import Pinecone
        from vector_store import index_documents
        def _progress(msg):
            if progress_callback:
                try:
                    progress_callback(msg)
                except Exception:
                    pass
        _progress("Loading client & Pinecone index...")
        client, _, embedding_model = config.get_rag_client()
        # Prefer pipeline's index getter so the index is created if missing
        try:
            from rag_pipeline import _get_pinecone_index
            index = _get_pinecone_index(skip_ensure=False)
        except Exception:
            pc = Pinecone(api_key=config.PINECONE_API_KEY)
            index = pc.Index(config.PINECONE_INDEX)
        data_dir = Path(config.DATA_DIR).resolve()
        if not data_dir.is_dir():
            _progress("No data/ directory.")
            return 0
        _progress("Indexing documents from data/...")
        return index_documents(
            index, client, embedding_model, data_dir,
            namespace=config.PINECONE_NAMESPACE_DOCS,
            chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP,
            progress_callback=progress_callback,
            embedding_timeout=getattr(config, "OPENAI_TIMEOUT", 60.0),
        )

from check_connections import run_checks as run_connection_checks


def main():
    st.set_page_config(page_title="Hybrid Graph RAG", page_icon="🔗", layout="centered")
    st.title("Hybrid Graph + Vector RAG")
    st.caption("Entity extraction → Graph (Neo4j) + Vector (Pinecone) → LLM answer.")

    with st.expander("Index documents (run once so answers use your data)", expanded=True):
        import config as _cfg
        _data_dir = getattr(_cfg, "DATA_DIR", None)
        _data_path = str(_data_dir.resolve() if _data_dir and hasattr(_data_dir, "resolve") else _data_dir)
        st.caption(f"Index files from **data/** into Pinecone. Path: `{_data_path}`. Add .txt or .md there, then click below. Indexing uses parallel API calls for speed.")
        st.caption("**Fastest first run:** run in terminal before opening the app (no timeout):")
        st.code("cd RAGs/HybridGraph_RAG && python index_docs.py", language="bash")
        if st.button("Index documents now"):
            idx_status = st.empty()
            try:
                with idx_status.status("Indexing...", expanded=True) as status:
                    def progress(msg):
                        status.update(label=msg, state="running")
                    n = run_index_documents(progress_callback=progress)
                    status.update(label=f"Done: indexed {n} chunks.", state="complete")
                idx_status.empty()
                st.success(f"Indexed {n} chunks. You can now ask questions about your data.")
            except Exception as e:
                idx_status.empty()
                st.error(str(e))

    with st.expander("Test connections (Neo4j, Pinecone, OpenAI/Azure)", expanded=False):
        if st.button("Run connection check"):
            with st.spinner("Checking..."):
                import io
                from contextlib import redirect_stdout
                buf = io.StringIO()
                try:
                    with redirect_stdout(buf):
                        exit_code = run_connection_checks()
                    out = buf.getvalue()
                    if exit_code == 0:
                        st.success("All connections OK.")
                    else:
                        st.warning("Some checks failed. See output below.")
                    st.code(out, language="text")
                except Exception as e:
                    st.error(str(e))

    query = st.text_area("Question", placeholder="Ask a question...", height=100)
    if st.button("Get answer"):
        if not (query or "").strip():
            st.warning("Enter a question.")
            return
        status_placeholder = st.empty()
        try:
            with status_placeholder.status("Running pipeline...", expanded=True) as status:
                def progress(msg):
                    status.update(label=msg, state="running")

                answer = run(
                    query.strip(),
                    index_docs_if_empty=False,
                    skip_pinecone_ensure=True,
                    progress_callback=progress,
                )
                status.update(label="Done.", state="complete")
            status_placeholder.empty()
            st.divider()
            st.subheader("Answer")
            st.write(answer)
        except Exception as e:
            status_placeholder.empty()
            st.error(f"Error: {e}")
            st.caption(
                "Run **Test connections** to see which service fails. "
                "If answers lack your data: open **Index documents** above and click **Index documents now** (run once)."
            )


if __name__ == "__main__":
    main()
