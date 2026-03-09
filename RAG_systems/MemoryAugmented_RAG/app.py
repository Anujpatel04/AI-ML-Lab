"""
Memory-Augmented RAG — Streamlit frontend.
Run: streamlit run app.py (use repo root .venv so pinecone is available).
"""
import os
import sys

# Ensure we run with repo root .venv (where pinecone is installed)
def _ensure_venv():
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
    raise ImportError(
        "pinecone not found. Use repo root .venv: source .venv/bin/activate then streamlit run app.py"
    )

_ensure_venv()

import streamlit as st

from rag_pipeline import run


def main():
    st.set_page_config(page_title="Memory RAG", page_icon="🧠", layout="centered")
    st.title("Memory-Augmented RAG")
    st.caption("Ask a question. Context from memory + documents is used and the turn is stored.")

    query = st.text_area("Question", placeholder="Ask anything...", height=120)
    if st.button("Get answer"):
        if not (query or "").strip():
            st.warning("Enter a question.")
            return
        with st.spinner("Searching memory & docs, generating answer..."):
            try:
                answer = run(query.strip())
                st.divider()
                st.subheader("Answer")
                st.write(answer)
            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
