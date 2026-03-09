"""
Streamlit frontend for Multi-Agent RAG.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
from core.orchestrator import Orchestrator, PipelineResult
from core.vector_store import build_index

st.set_page_config(page_title="Multi-Agent RAG", page_icon="🤖", layout="centered")
st.title("Multi-Agent RAG")
st.caption("Retriever → Reasoning → Verification pipeline. Ask a question to get a validated answer with sources.")

# Sidebar
with st.sidebar:
    st.header("Settings")
    top_k = st.slider("Top-k documents", min_value=1, max_value=20, value=5)
    show_stages = st.checkbox("Expandable stages", value=True)
    st.divider()
    st.caption("After adding or changing files in data/documents/, rebuild the index.")
    if st.button("Rebuild index"):
        with st.spinner("Building index from data/documents/..."):
            try:
                n = build_index()
                st.success(f"Index rebuilt: {n} chunks from all documents.")
            except Exception as e:
                st.error(str(e))

query = st.text_area("Question", placeholder="Ask a question...", height=100)
if st.button("Get answer", type="primary"):
    if not (query or "").strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Running pipeline..."):
            try:
                orch = Orchestrator(top_k=top_k)
                result: PipelineResult = orch.run(query.strip())

                st.subheader("Final answer")
                st.write(result.final_answer)

                if show_stages:
                    with st.expander("Retrieved documents"):
                        if not result.retrieved_docs:
                            st.info("No documents retrieved.")
                        else:
                            for i, doc in enumerate(result.retrieved_docs, 1):
                                score = doc.get("score", 0)
                                content = doc.get("content", "")
                                st.markdown(f"**{i}** (score: {score:.3f})")
                                st.text(content[:500] + ("..." if len(content) > 500 else ""))

                    with st.expander("Reasoning output"):
                        st.write(result.reasoning_output or "(none)")

                    with st.expander("Verification"):
                        status = "Passed" if result.verified else "Refined / Failed"
                        st.write(f"**Status:** {status}")
                        st.write(result.verification_reason)
                        if result.refined:
                            st.caption("Answer was refined to better match the sources.")

                st.caption("Sources: " + str(len(result.retrieved_docs)) + " document(s) used.")
            except Exception as e:
                st.error(str(e))
