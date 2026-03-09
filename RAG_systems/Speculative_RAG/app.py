"""
Speculative RAG: Streamlit UI.
Run with: streamlit run app.py
"""

import logging
import sys

import streamlit as st

import config
from retriever.retriever import Retriever
from pipeline.speculative_rag_pipeline import SpeculativeRAGPipeline, SpeculativeRAGResult

# Logging to stderr so Streamlit is not cluttered
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)


def get_pipeline() -> SpeculativeRAGPipeline:
    """Build and return the speculative RAG pipeline (client, retriever, models from config)."""
    client, draft_model, verifier_model, embedding_model = config.get_rag_client()
    retriever = Retriever(
        data_dir=config.DATA_DIR,
        index_path=config.INDEX_PATH,
        chunks_path=config.CHUNKS_PATH,
        client=client,
        embedding_model=embedding_model,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return SpeculativeRAGPipeline(
        retriever=retriever,
        client=client,
        draft_model=draft_model,
        verifier_model=verifier_model,
        top_k=config.TOP_K,
        temperature=config.TEMPERATURE,
        max_tokens=config.MAX_TOKENS,
    )


def render_result(result: SpeculativeRAGResult) -> None:
    """Render pipeline result in the UI."""
    st.subheader("Retrieved documents")
    for i, chunk in enumerate(result.context_chunks, 1):
        with st.expander(f"Chunk {i}"):
            st.text(chunk[:2000] + ("..." if len(chunk) > 2000 else ""))

    st.subheader("Draft answer (small model)")
    st.write(result.draft_answer)

    st.subheader("Verification feedback")
    st.info(result.verification_feedback)

    st.subheader("Final answer (verified)")
    st.success(result.final_answer)


def main():
    st.set_page_config(page_title="Speculative RAG", page_icon="📚", layout="wide")
    st.title("Speculative RAG")
    st.caption(
        "Retrieve documents → small model drafts an answer → large model verifies and improves."
    )

    if "pipeline" not in st.session_state:
        try:
            st.session_state.pipeline = get_pipeline()
        except ValueError as e:
            st.error(str(e))
            st.stop()

    question = st.text_area("Question", placeholder="Ask a question based on your documents...", height=100)
    if st.button("Run speculative RAG"):
        if not (question or "").strip():
            st.warning("Please enter a question.")
            return
        with st.spinner("Retrieving → Drafting → Verifying..."):
            try:
                result = st.session_state.pipeline.run(question.strip())
                render_result(result)
            except Exception as e:
                logger.exception("Pipeline error: %s", e)
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()
