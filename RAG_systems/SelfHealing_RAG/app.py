"""
Self-Healing RAG Assistant – Streamlit frontend.
Detects poor retrieval and retries before generating the final answer.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import streamlit as st
import config
from backend.pipeline import (
    retrieve_documents,
    evaluate_relevance,
    retry_retrieval,
    generate_answer,
)

st.set_page_config(
    page_title="Self-Healing RAG Assistant",
    page_icon="🔄",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Sidebar
with st.sidebar:
    st.header("Self-Healing RAG Assistant")
    st.caption("Retrieval → Relevance check → Optional retry → Answer")
    st.divider()

    st.subheader("Settings")
    chat_options = ["gpt-4o", "gpt-4o-mini", "gpt-35-turbo", config.AZURE_CHAT_DEPLOYMENT]
    chat_options = list(dict.fromkeys(chat_options))
    llm_model = st.selectbox(
        "LLM (chat)",
        options=chat_options,
        index=0 if chat_options else 0,
        help="Azure OpenAI chat deployment",
    )
    embed_options = ["text-embedding-ada-002", "text-embedding-3-small", config.AZURE_EMBEDDING_DEPLOYMENT]
    embed_options = list(dict.fromkeys(embed_options))
    embedding_model = st.selectbox(
        "Embedding model",
        options=embed_options,
        index=0 if embed_options else 0,
        help="Azure OpenAI embedding deployment",
    )
    top_k = st.slider("Top-k documents", min_value=1, max_value=20, value=5)
    relevance_threshold = st.slider(
        "Relevance threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="LLM must judge retrieval above this to skip retry",
    )
    enable_self_healing = st.toggle("Enable Self-Healing Retrieval", value=True)

    st.divider()
    st.caption("Model: Azure OpenAI (LLM + embeddings). Pinecone for retrieval. Configure in repo root .env.")

# Main
st.title("Self-Healing RAG Assistant")
st.caption("Ask a question. The system retrieves documents, evaluates relevance, and may retry retrieval before answering.")

query = st.text_area("Question", placeholder="Enter your question...", height=120)
submitted = st.button("Submit", type="primary")

if submitted and (query or "").strip():
    query = query.strip()
    docs = []
    relevance_ok = False
    relevance_message = ""
    retry_triggered = False
    retry_log = []
    final_docs = []

    with st.spinner("Running pipeline..."):
        # Step 1: Retrieval
        with st.status("Retrieval", state="running") as status:
            docs = retrieve_documents(
                query, top_k=top_k, embedding_deployment=embedding_model
            )
            if docs:
                status.update(label="Retrieval complete", state="complete")
            else:
                status.update(label="Retrieval returned no documents", state="complete")

        # Step 2: Relevance check
        with st.status("Relevance check", state="running") as status:
            relevance_ok, relevance_message = evaluate_relevance(
                query, docs, threshold=relevance_threshold, chat_deployment=llm_model
            )
            status.update(
                label="Relevance check complete",
                state="complete",
            )

        # Step 3: Retry if self-healing enabled and relevance low
        if enable_self_healing and not relevance_ok and docs:
            with st.status("Retry retrieval", state="running") as status:
                retry_log.append("Initial retrieval quality low. Performing secondary search…")
                final_docs = retry_retrieval(
                    query, top_k=min(top_k * 2, 20), embedding_deployment=embedding_model
                )
                retry_triggered = True
                if final_docs:
                    retry_log.append(f"Secondary search returned {len(final_docs)} documents.")
                    status.update(label="Retry retrieval complete", state="complete")
                else:
                    retry_log.append("Secondary search returned no additional documents.")
                    final_docs = docs
                    status.update(label="Retry retrieval complete", state="complete")
        else:
            final_docs = docs

        # Step 4: Final answer
        with st.status("Final answer", state="running") as status:
            answer = generate_answer(query, final_docs, chat_deployment=llm_model)
            status.update(label="Final answer ready", state="complete")

    # Display: pipeline steps summary
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Retrieval", f"{len(docs)} docs", delta=None)
    with c2:
        st.metric("Relevance", "Pass" if relevance_ok else "Low", delta=None)
    with c3:
        st.metric("Retry", "Yes" if retry_triggered else "No", delta=None)
    with c4:
        st.metric("Final docs", str(len(final_docs)), delta=None)

    # Expandable: Retrieved documents
    with st.expander("Retrieved documents", expanded=False):
        if not final_docs:
            st.info(
                "No documents retrieved. Check: (1) Pinecone index **memory-rag** has vectors in namespace **documents**, "
                "(2) repo .env has PINECONE_API_KEY, (3) if using serverless, try setting PINECONE_HOST in .env to the index host from Pinecone console."
            )
        else:
            for i, d in enumerate(final_docs, 1):
                score = d.get("score", 0)
                content = d.get("content", "")
                st.markdown(f"**{i}** (score: `{score:.3f}`)")
                st.text(content[:600] + ("…" if len(content) > 600 else ""))

    # Expandable: Relevance evaluation
    with st.expander("LLM relevance evaluation", expanded=False):
        st.write("**Result:** " + ("Sufficient" if relevance_ok else "Insufficient"))
        st.write(relevance_message)

    # Expandable: Retry log
    if retry_log:
        with st.expander("Retrieval retry status", expanded=True):
            for line in retry_log:
                st.write(line)

    # Final answer
    st.subheader("Final answer")
    st.write(answer)

elif submitted:
    st.warning("Please enter a question.")
