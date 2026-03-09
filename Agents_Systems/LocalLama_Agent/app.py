import os
import re
import tempfile
import inspect
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import bs4
import streamlit as st
from agno.agent import Agent
from agno.models.ollama import Ollama
from agno.tools.exa import ExaTools
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
try:
    from langchain_ollama import OllamaEmbeddings  # type: ignore
except Exception:  # pragma: no cover
    from langchain_community.embeddings import OllamaEmbeddings  # type: ignore

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

ROOT_DIR = Path(__file__).resolve().parents[2]
load_dotenv(ROOT_DIR / ".env")


COLLECTION_NAME = "local-rag"
LLM_MODEL_ID = "llama3.1:latest"
BUNDLED_PDF_NAME = "Resume__Anuj.pdf"


def init_state() -> None:
    defaults = {
        "qdrant_api_key": os.getenv("QDRANT_API_KEY", ""),
        "qdrant_url": os.getenv("QDRANT_URL", ""),
        "exa_api_key": os.getenv("EXA_API_KEY", ""),
        "embed_model": os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
        "use_web_search": False,
        "force_web_search": False,
        "similarity_threshold": 0.7,
        "rag_enabled": True,
        "vector_store": None,
        "processed_documents": [],
        "history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def init_qdrant() -> Optional[QdrantClient]:
    if not (st.session_state.qdrant_url and st.session_state.qdrant_api_key):
        return None
    try:
        return QdrantClient(url=st.session_state.qdrant_url, api_key=st.session_state.qdrant_api_key, timeout=60)
    except Exception as e:
        st.error(f"Qdrant connection failed: {e}")
        return None


def process_pdf(file) -> List:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(file.getvalue())
            loader = PyPDFLoader(tmp_file.name)
            documents = loader.load()

        for doc in documents:
            doc.metadata.update(
                {
                    "source_type": "pdf",
                    "file_name": file.name,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(documents)
    except Exception as e:
        st.error(f"PDF processing error: {e}")
        return []


def process_pdf_path(pdf_path: Path) -> List:
    """Process a local PDF file path and add source metadata."""
    try:
        loader = PyPDFLoader(str(pdf_path))
        documents = loader.load()

        for doc in documents:
            doc.metadata.update(
                {
                    "source_type": "pdf",
                    "file_name": pdf_path.name,
                    "timestamp": datetime.now().isoformat(),
                    "local_path": str(pdf_path),
                }
            )

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(documents)
    except Exception as e:
        st.error(f"Local PDF processing error: {e}")
        return []


def process_web(url: str) -> List:
    try:
        loader = WebBaseLoader(
            web_paths=(url,),
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(class_=("post-content", "post-title", "post-header", "content", "main"))
            ),
        )
        documents = loader.load()

        for doc in documents:
            doc.metadata.update(
                {
                    "source_type": "url",
                    "url": url,
                    "timestamp": datetime.now().isoformat(),
                }
            )

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_documents(documents)
    except Exception as e:
        st.error(f"Web processing error: {e}")
        return []


def create_or_get_vector_store(client: QdrantClient) -> Optional[QdrantVectorStore]:
    try:
        embedder = OllamaEmbeddings(model=st.session_state.embed_model)
        embed_dim = len(embedder.embed_query("dimension_check"))

        try:
            client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=embed_dim, distance=Distance.COSINE),
            )
            st.success(f"Created Qdrant collection: {COLLECTION_NAME}")
        except Exception as e:
            if "already exists" not in str(e).lower():
                raise

        return QdrantVectorStore(client=client, collection_name=COLLECTION_NAME, embedding=embedder)
    except Exception as e:
        st.error(f"Vector store error: {e}")
        return None


def get_web_search_agent(search_domains: List[str]) -> Agent:
    agent_kwargs = {
        "tools": [
            ExaTools(
                api_key=st.session_state.exa_api_key,
                include_domains=search_domains,
                num_results=5,
            )
        ],
        "instructions": (
            "Search the web for relevant information about the query, then summarize the most relevant points. "
            "Include sources in your response."
        ),
        "show_tool_calls": True,
        "markdown": True,
    }
    supported = set(inspect.signature(Agent.__init__).parameters.keys())
    agent_kwargs = {k: v for k, v in agent_kwargs.items() if k in supported}

    return Agent(
        name="Web Search Agent",
        model=Ollama(id=LLM_MODEL_ID),
        **agent_kwargs,
    )


def get_rag_agent() -> Agent:
    agent_kwargs = {
        "instructions": (
            "You are an assistant that answers questions accurately.\n"
            "- If context is provided, prioritize it.\n"
            "- If web search results are provided, clearly indicate they came from web search.\n"
            "- Be clear and concise."
        ),
        "show_tool_calls": True,
        "markdown": True,
    }
    supported = set(inspect.signature(Agent.__init__).parameters.keys())
    agent_kwargs = {k: v for k, v in agent_kwargs.items() if k in supported}

    return Agent(
        name="Local RAG Agent",
        model=Ollama(id=LLM_MODEL_ID),
        **agent_kwargs,
    )


def strip_think_tags(text: str) -> Tuple[Optional[str], str]:
    think_pattern = r"<think>(.*?)</think>"
    m = re.search(think_pattern, text, re.DOTALL)
    if not m:
        return None, text
    thinking = m.group(1).strip()
    final = re.sub(think_pattern, "", text, flags=re.DOTALL).strip()
    return thinking, final


def main() -> None:
    init_state()

    st.title("Local RAG (Ollama)")
    st.caption("Local chat with optional RAG (Qdrant) and optional web-search fallback (Exa).")

    top_left, top_right = st.columns([0.75, 0.25])
    with top_left:
        st.info(f"Using local Ollama model: `{LLM_MODEL_ID}`")
    with top_right:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    with st.expander("RAG & Search Settings", expanded=True):
        st.session_state.rag_enabled = st.toggle("Enable RAG mode", value=st.session_state.rag_enabled)

        st.session_state.similarity_threshold = st.slider(
            "Similarity threshold",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.similarity_threshold),
            help="Higher is stricter; lower returns more chunks.",
            disabled=not st.session_state.rag_enabled,
        )

        st.session_state.use_web_search = st.checkbox(
            "Enable web search fallback (Exa)",
            value=st.session_state.use_web_search,
        )

        st.session_state.exa_api_key = os.getenv("EXA_API_KEY", st.session_state.exa_api_key)
        if st.session_state.use_web_search:
            if st.session_state.exa_api_key:
                st.success("EXA_API_KEY detected from root .env")
            else:
                st.warning("EXA_API_KEY not found in root .env (web fallback will be disabled).")

        default_domains = ["arxiv.org", "wikipedia.org", "github.com", "medium.com"]
        custom_domains = st.text_input("Web search domains (comma-separated)", value=",".join(default_domains))
        search_domains: List[str] = [d.strip() for d in custom_domains.split(",") if d.strip()]

    qdrant_client: Optional[QdrantClient] = None
    if st.session_state.rag_enabled:
        st.session_state.qdrant_url = os.getenv("QDRANT_URL", st.session_state.qdrant_url)
        st.session_state.qdrant_api_key = os.getenv("QDRANT_API_KEY", st.session_state.qdrant_api_key)
        st.session_state.embed_model = os.getenv("OLLAMA_EMBED_MODEL", st.session_state.embed_model)

        with st.expander("Data Upload (RAG)", expanded=True):
            if st.session_state.qdrant_url and st.session_state.qdrant_api_key:
                st.success("Qdrant config detected from root .env")
                st.caption(f"Collection: `{COLLECTION_NAME}` | Embeddings: `{st.session_state.embed_model}`")
            else:
                st.warning("Qdrant config missing in root .env (RAG storage/retrieval will be disabled).")

            qdrant_client = init_qdrant()
            if qdrant_client and st.session_state.vector_store is None:
                st.session_state.vector_store = create_or_get_vector_store(qdrant_client)

            bundled_pdf_path = Path(__file__).resolve().parent / BUNDLED_PDF_NAME
            if bundled_pdf_path.exists():
                if st.button(f"Index bundled PDF: {BUNDLED_PDF_NAME}", use_container_width=True):
                    if not st.session_state.vector_store:
                        st.error("Vector store not ready. Check Qdrant config in root .env.")
                    else:
                        source_key = f"local:{BUNDLED_PDF_NAME}"
                        if source_key in st.session_state.processed_documents:
                            st.info("Already indexed.")
                        else:
                            with st.spinner(f"Indexing {BUNDLED_PDF_NAME}..."):
                                chunks = process_pdf_path(bundled_pdf_path)
                                if chunks:
                                    st.session_state.vector_store.add_documents(chunks)
                                    st.session_state.processed_documents.append(source_key)
                                    st.success(f"Indexed: {BUNDLED_PDF_NAME}")
            else:
                st.caption(f"Bundled PDF not found: {BUNDLED_PDF_NAME}")

            uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
            web_url = st.text_input("Or enter URL")

            if uploaded_file and uploaded_file.name not in st.session_state.processed_documents:
                with st.spinner("Processing PDF..."):
                    chunks = process_pdf(uploaded_file)
                    if chunks and st.session_state.vector_store:
                        st.session_state.vector_store.add_documents(chunks)
                        st.session_state.processed_documents.append(uploaded_file.name)
                        st.success(f"Added: {uploaded_file.name}")

            if web_url and web_url not in st.session_state.processed_documents:
                with st.spinner("Processing URL..."):
                    chunks = process_web(web_url)
                    if chunks and st.session_state.vector_store:
                        st.session_state.vector_store.add_documents(chunks)
                        st.session_state.processed_documents.append(web_url)
                        st.success(f"Added: {web_url}")

            if st.session_state.processed_documents:
                st.subheader("Sources")
                for src in st.session_state.processed_documents:
                    st.write(f"- {src}")
    else:
        default_domains = ["arxiv.org", "wikipedia.org", "github.com", "medium.com"]
        search_domains = default_domains

    chat_col, toggle_col = st.columns([0.9, 0.1])
    with chat_col:
        prompt = st.chat_input("Ask about your documents..." if st.session_state.rag_enabled else "Ask me anything...")
    with toggle_col:
        st.session_state.force_web_search = st.toggle("Web", help="Force web search")

    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if not prompt:
        if st.session_state.rag_enabled:
            st.info("Upload a PDF or URL, then ask a question.")
        else:
            st.info("Ask a question to chat locally with Llama.")
        return

    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    context = ""
    docs = []

    if st.session_state.rag_enabled and not st.session_state.force_web_search and st.session_state.vector_store:
        retriever = st.session_state.vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 5, "score_threshold": float(st.session_state.similarity_threshold)},
        )
        with st.spinner("Searching your documents..."):
            docs = retriever.invoke(prompt)
        if docs:
            context = "\n\n".join([d.page_content for d in docs])
            st.info(f"Found {len(docs)} relevant chunks (threshold {st.session_state.similarity_threshold}).")

    if (st.session_state.force_web_search or not context) and st.session_state.use_web_search:
        if not st.session_state.exa_api_key:
            st.warning("Web search is enabled, but EXA_API_KEY is missing.")
        else:
            with st.spinner("Searching the web..."):
                web_agent = get_web_search_agent(search_domains)
                web_results = web_agent.run(prompt).content
                if web_results:
                    context = f"Web Search Results:\n{web_results}"

    with st.spinner("Thinking..."):
        rag_agent = get_rag_agent()
        if context:
            full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer using the context. Be clear and accurate."
        else:
            full_prompt = prompt
        response = rag_agent.run(full_prompt).content

    thinking, final = strip_think_tags(response)

    st.session_state.history.append({"role": "assistant", "content": final})
    with st.chat_message("assistant"):
        if thinking:
            with st.expander("Show reasoning"):
                st.markdown(thinking)
        st.markdown(final)

        if docs and not st.session_state.force_web_search and st.session_state.rag_enabled:
            with st.expander("Sources (document chunks)"):
                for i, doc in enumerate(docs, 1):
                    source_type = doc.metadata.get("source_type", "unknown")
                    src = doc.metadata.get("file_name") if source_type == "pdf" else doc.metadata.get("url")
                    st.write(f"{i}. {source_type}: {src}")
                    st.write(doc.page_content[:300] + "...")


if __name__ == "__main__":
    main()


