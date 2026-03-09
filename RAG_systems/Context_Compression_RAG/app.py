"""
Context Compression RAG: retrieve top 20 chunks, LLM compress to top 5, then generate answer.
Single production-ready script with Streamlit frontend. Azure OpenAI from repo root .env.
"""

import logging
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DOTENV_PATH = REPO_ROOT / ".env"

if DOTENV_PATH.is_file():
    from dotenv import load_dotenv
    load_dotenv(DOTENV_PATH)

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO")),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Azure config: support both AZURE_OPENAI_* and AZURE_* / AZURE_ENDPOINT
AZURE_API_KEY = (
    os.environ.get("AZURE_OPENAI_API_KEY")
    or os.environ.get("AZURE_KEY")
    or ""
).strip()
AZURE_ENDPOINT_RAW = (
    os.environ.get("AZURE_OPENAI_ENDPOINT")
    or os.environ.get("AZURE_ENDPOINT")
    or ""
).strip()
AZURE_API_VERSION = (
    os.environ.get("AZURE_OPENAI_API_VERSION")
    or os.environ.get("API_VERSION")
    or "2024-02-15-preview"
).strip()


def _parse_azure_endpoint(url: str) -> Tuple[str, str]:
    """Extract base URL and deployment name from endpoint URL if present."""
    url = (url or "").strip()
    if "/openai/deployments/" in url:
        base = url.split("/openai/deployments/")[0].strip("/")
        rest = url.split("/openai/deployments/")[1]
        deployment = rest.split("/")[0].split("?")[0].strip() if rest else "gpt-4o"
        return base, deployment
    return url.strip("/"), "gpt-4o"


AZURE_BASE_URL, AZURE_CHAT_DEPLOYMENT = _parse_azure_endpoint(AZURE_ENDPOINT_RAW)
AZURE_EMBEDDING_DEPLOYMENT = (
    os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    or os.environ.get("AZURE_EMBEDDING_DEPLOYMENT")
    or "text-embedding-ada-002"
).strip()

TOP_K_RETRIEVE = 20
TOP_K_AFTER_COMPRESSION = 5
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBEDDING_DIM = 1536


def _get_client():
    """Return Azure OpenAI client. Raises if config missing."""
    if not AZURE_API_KEY or not AZURE_BASE_URL:
        raise ValueError(
            "Set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT (or AZURE_KEY and AZURE_ENDPOINT) in repo root .env"
        )
    from openai import AzureOpenAI
    return AzureOpenAI(
        api_key=AZURE_API_KEY,
        azure_endpoint=AZURE_BASE_URL,
        api_version=AZURE_API_VERSION,
    )


def load_documents(file_contents: List[Tuple[str, bytes]]) -> List[str]:
    """
    Load text from uploaded PDF bytes. Each item is (filename, raw_bytes).
    Returns list of full-document text strings.
    """
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError(
            "pypdf is required for PDF uploads. Install with: pip install pypdf "
            "(or use a venv: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt)"
        ) from None
    import io
    out: List[str] = []
    for name, raw in file_contents:
        if not raw or not name.lower().endswith(".pdf"):
            continue
        try:
            reader = PdfReader(io.BytesIO(raw))
            parts = []
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    parts.append(t)
            if parts:
                out.append("\n".join(parts))
        except Exception as e:
            logger.warning("Failed to load %s: %s", name, e)
    return out


def split_documents(documents: List[str], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split documents into overlapping text chunks. Always advances to avoid infinite loop."""
    if overlap >= chunk_size:
        overlap = chunk_size // 2
    chunks: List[str] = []
    for doc in documents:
        doc = (doc or "").replace("\r\n", "\n").strip()
        if not doc:
            continue
        start = 0
        doc_chunk_count = 0
        max_chunks_per_doc = max(5000, (len(doc) // max(1, chunk_size - overlap)) + 2)
        while start < len(doc):
            if doc_chunk_count >= max_chunks_per_doc:
                break
            end = min(start + chunk_size, len(doc))
            if end < len(doc):
                br = doc.rfind("\n", start, end + 1)
                if br > start:
                    end = br + 1
                else:
                    br = doc.rfind(" ", start, end + 1)
                    if br > start:
                        end = br + 1
            block = doc[start:end].strip()
            if block:
                chunks.append(block)
                doc_chunk_count += 1
            if end >= len(doc):
                break
            prev_start = start
            start = end - overlap
            if start <= 0 or start <= prev_start:
                start = end
    return chunks


def create_embeddings(client, texts: List[str], deployment: str = AZURE_EMBEDDING_DEPLOYMENT) -> List[List[float]]:
    """Create embeddings for a list of texts via Azure OpenAI. Batches of 100."""
    out: List[List[float]] = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            resp = client.embeddings.create(input=batch, model=deployment)
            for item in resp.data:
                out.append(item.embedding)
        except Exception as e:
            logger.exception("Embeddings batch failed")
            raise RuntimeError(f"Embeddings failed: {e}") from e
    return out


def build_vector_store(chunks: List[str], embeddings: List[List[float]]):
    """Build and return a FAISS index and the chunk list for later retrieval."""
    import numpy as np
    try:
        import faiss
    except ImportError:
        raise ImportError("Install faiss-cpu: pip install faiss-cpu") from None
    if not chunks or not embeddings or len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings length must match and be non-empty")
    dim = len(embeddings[0])
    matrix = np.array(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(matrix)
    index.add(matrix)
    return index, chunks, matrix


def retrieve_documents(
    query: str,
    client,
    index,
    chunk_list: List[str],
    top_k: int = TOP_K_RETRIEVE,
    embedding_deployment: str = AZURE_EMBEDDING_DEPLOYMENT,
) -> List[str]:
    """Embed query, search FAISS, return top_k chunk texts."""
    import numpy as np
    try:
        import faiss
    except ImportError:
        raise ImportError("faiss-cpu is required: pip install faiss-cpu") from None
    if not query or not chunk_list:
        return []
    emb = create_embeddings(client, [query.strip()], deployment=embedding_deployment)
    if not emb:
        return []
    q = np.array([emb[0]], dtype=np.float32)
    faiss.normalize_L2(q)
    scores, indices = index.search(q, min(top_k, len(chunk_list)))
    out: List[str] = []
    for idx in indices[0]:
        if 0 <= idx < len(chunk_list):
            out.append(chunk_list[idx])
    return out


def compress_context(
    client,
    query: str,
    chunks: List[str],
    top_k: int = TOP_K_AFTER_COMPRESSION,
    deployment: str = AZURE_CHAT_DEPLOYMENT,
) -> List[str]:
    """
    Use LLM to rank and compress chunks; return top_k compressed/summarized sections.
    """
    if not chunks or not query:
        return []
    prompt = """You are a context compression system.
Given the following retrieved document chunks and a user question, remove redundant information and keep only the most relevant information needed to answer the question.
Output exactly """ + str(top_k) + """ compressed context sections, one per block. Each block must be a concise summary of the most relevant content from one or more chunks.
Format: number each section (1., 2., ...). Do not include the user question in the output. Only output the """ + str(top_k) + """ compressed sections."""

    chunk_block = "\n\n---\n\n".join(f"[Chunk {i+1}]\n{c}" for i, c in enumerate(chunks[:20]))
    user_content = f"User question: {query}\n\nRetrieved chunks:\n\n{chunk_block}"

    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            max_tokens=2048,
        )
        text = (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.exception("Compression API failed")
        raise RuntimeError(f"Context compression failed: {e}") from e

    sections: List[str] = []
    for part in re.split(r"\n\s*\d+\.\s*", text):
        part = part.strip()
        if part:
            sections.append(part)
    if len(sections) > top_k:
        sections = sections[:top_k]
    while len(sections) < top_k and chunks:
        sections.append(chunks[len(sections) % len(chunks)][:500])
    return sections[:top_k]


def generate_answer(
    client,
    query: str,
    context_sections: List[str],
    deployment: str = AZURE_CHAT_DEPLOYMENT,
) -> str:
    """Generate final answer from query and compressed context."""
    context_block = "\n\n".join(context_sections)
    prompt = """Answer the user's question using only the following context. If the context does not contain enough information, say so. Be concise and accurate."""

    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {query}\n\nAnswer:"},
            ],
            temperature=0.3,
            max_tokens=1024,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        logger.exception("Answer generation failed")
        raise RuntimeError(f"Answer generation failed: {e}") from e


def run_pipeline(
    query: str,
    index,
    chunk_list: List[str],
    client,
) -> Tuple[List[str], List[str], str]:
    """
    Run: retrieve top 20 -> compress to top 5 -> generate answer.
    Returns (retrieved_chunks, compressed_sections, answer).
    """
    retrieved = retrieve_documents(
        query, client, index, chunk_list, top_k=TOP_K_RETRIEVE
    )
    compressed = compress_context(client, query, retrieved, top_k=TOP_K_AFTER_COMPRESSION)
    answer = generate_answer(client, query, compressed)
    return retrieved, compressed, answer


# --- Streamlit ---

def main():
    import streamlit as st

    st.set_page_config(page_title="Context Compression RAG", layout="centered")
    st.title("Context Compression RAG Demo")

    try:
        client = _get_client()
    except ValueError as e:
        st.error(str(e))
        st.caption("Configure Azure in repo root .env: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT (or AZURE_KEY, AZURE_ENDPOINT).")
        return

    uploaded = st.file_uploader("Upload PDF documents", type=["pdf"], accept_multiple_files=True)
    index_key = "faiss_index"
    chunks_key = "faiss_chunks"

    if uploaded and (index_key not in st.session_state or not st.session_state.get(chunks_key)):
        with st.spinner("Loading and indexing documents..."):
            try:
                file_contents = [(f.name, f.read()) for f in uploaded]
                docs = load_documents(file_contents)
                if not docs:
                    st.warning("No text could be extracted from the uploaded PDFs.")
                else:
                    chunks = split_documents(docs)
                    if not chunks:
                        st.warning("No chunks produced from documents.")
                    else:
                        emb = create_embeddings(client, chunks)
                        idx, clist, _ = build_vector_store(chunks, emb)
                        st.session_state[index_key] = idx
                        st.session_state[chunks_key] = clist
                        st.success(f"Indexed {len(chunks)} chunks from {len(docs)} document(s).")
            except Exception as e:
                st.error(str(e))
                logger.exception("Indexing failed")

    query = st.text_input("Query", placeholder="Ask a question based on the uploaded documents.")
    if st.button("Get answer"):
        if not (query or "").strip():
            st.warning("Enter a query.")
        elif index_key not in st.session_state:
            st.warning("Upload and process PDFs first.")
        else:
            with st.spinner("Retrieving, compressing, and generating answer..."):
                try:
                    idx = st.session_state[index_key]
                    clist = st.session_state[chunks_key]
                    retrieved, compressed, answer = run_pipeline(query.strip(), idx, clist, client)
                    st.subheader("Retrieved documents (top 20)")
                    st.text_area("", value="\n\n---\n\n".join(retrieved), height=200, disabled=True, label_visibility="collapsed")
                    st.subheader("Compressed context (top 5)")
                    st.text_area("", value="\n\n".join(compressed), height=200, disabled=True, label_visibility="collapsed")
                    st.subheader("Final answer")
                    st.write(answer)
                except Exception as e:
                    st.error(str(e))
                    logger.exception("Pipeline failed")


if __name__ == "__main__":
    main()
