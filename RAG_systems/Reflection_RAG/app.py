"""
Reflection RAG - Streamlit UI.
Run: streamlit run app.py
"""

import logging
from pathlib import Path

import streamlit as st

# Ensure project root is on path when running as streamlit run app.py
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(_project_root))

import config
from pipeline import RAGPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_pipeline() -> RAGPipeline:
    """Build pipeline; uses Azure (AZURE_ENDPOINT, AZURE_KEY, API_VERSION) or OPENAI_API_KEY from repo root .env."""
    return RAGPipeline(
        data_dir=config.DATA_DIR,
        index_path=config.INDEX_PATH,
        chunks_path=config.CHUNKS_PATH,
        top_k=config.TOP_K,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )


def main():
    st.set_page_config(page_title="Reflection RAG", layout="centered")
    st.title("Reflection RAG")
    st.caption("Retrieval -> Initial Answer -> LLM Critique -> Improved Answer")

    try:
        pipeline = get_pipeline()
    except ValueError as e:
        st.error(str(e))
        return

    query = st.text_area("Your question", height=100, placeholder="Ask a question based on your documents in data/...")

    if st.button("Run RAG", type="primary", use_container_width=True):
        if not (query or "").strip():
            st.warning("Enter a question.")
            return

        with st.spinner("Retrieving and generating..."):
            result = pipeline.run(query)

        if result.get("error"):
            st.error(result["error"])

        st.subheader("Retrieved context")
        chunks = result.get("context_chunks") or []
        if chunks:
            for i, c in enumerate(chunks, 1):
                with st.expander(f"Chunk {i}"):
                    st.text(c[:2000] + ("..." if len(c) > 2000 else ""))
        else:
            st.caption("No chunks retrieved.")

        st.subheader("Initial answer")
        st.write(result.get("initial_answer") or "(none)")

        st.subheader("Reflection critique")
        st.write(result.get("critique") or "(none)")

        st.subheader("Final improved answer")
        st.write(result.get("final_answer") or "(none)")


if __name__ == "__main__":
    main()
