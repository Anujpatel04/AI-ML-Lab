"""
Streamlit frontend for Adaptive RAG.
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*Pydantic V1.*Python 3.14.*")

_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import streamlit as st
from router import route_with_details

st.set_page_config(page_title="Adaptive RAG", page_icon="🔀", layout="centered")
st.title("Adaptive RAG")
st.caption("Query is classified as factual, entity, or broad; retrieval (vector / graph / hybrid) and answer are generated automatically.")

query = st.text_area("Question", placeholder="Ask a question...", height=100)
if st.button("Get answer", type="primary"):
    if not (query or "").strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Classifying and retrieving..."):
            try:
                out = route_with_details(query.strip())
                st.subheader("Answer")
                st.write(out["answer"])
                with st.expander("Retrieval: " + (out["type"] or "—")):
                    st.caption("Strategy: " + (out["type"] or "—"))
                    st.text_area("Context used", value=out["context"] or "", height=200, disabled=True)
            except Exception as e:
                st.error(str(e))
