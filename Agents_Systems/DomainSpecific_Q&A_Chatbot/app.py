import argparse
import json
import os
import sys
import time
import uuid
from typing import Callable, Dict, List, Optional, Tuple

import requests
import streamlit as st
import tiktoken
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone


ROOT_ENV_PATH = "/Users/anuj/Desktop/Anuj-AI-ML-Lab/.env"
load_dotenv(ROOT_ENV_PATH, override=True)

FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "question-answer"
PINECONE_INDEX_HOST = "https://question-answer-bq9e6pj.svc.aped-4627-b74a.pinecone.io"

DEFAULT_CHAT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
EMBED_DIMENSIONS = int(os.getenv("OPENAI_EMBED_DIMENSIONS", "1024"))

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5


SYSTEM_PROMPT = """You are a domain expert assistant.
Answer questions using only the provided context.
If the answer is not in the context, say: "I don't know based on my knowledge base."
Be accurate, concise, and professional.
Include source URLs when possible.
"""


def ensure_env() -> None:
    if not OPENAI_API_KEY:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    if not FIRECRAWL_API_KEY:
        raise EnvironmentError("FIRECRAWL_API_KEY is not set.")
    if not PINECONE_API_KEY:
        raise EnvironmentError("PINECONE_API_KEY is not set.")


def get_openai_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def slugify(text: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in text).strip("-")


def generate_queries(domain: str) -> List[str]:
    return [
        f"{domain} basics",
        f"{domain} fundamentals",
        f"{domain} best practices",
        f"{domain} key concepts",
        f"{domain} introduction",
    ]


def firecrawl_search(query: str, limit: int = 8) -> List[str]:
    url = "https://api.firecrawl.dev/v1/search"
    headers = {"Authorization": f"Bearer {FIRECRAWL_API_KEY}"}
    payload = {"query": query, "limit": limit}
    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    results = data.get("data", [])
    return [item.get("url") for item in results if item.get("url")]


def firecrawl_scrape(url: str) -> Tuple[str, str, str]:
    endpoint = "https://api.firecrawl.dev/v1/scrape"
    headers = {"Authorization": f"Bearer {FIRECRAWL_API_KEY}"}
    payload = {"url": url, "formats": ["markdown"], "onlyMainContent": True}
    response = requests.post(endpoint, headers=headers, json=payload, timeout=90)
    response.raise_for_status()
    data = response.json().get("data", {})

    text = data.get("markdown") or data.get("content") or ""
    title = data.get("title") or ""
    final_url = data.get("url") or url
    return text.strip(), title.strip(), final_url.strip()


def tokenize_chunks(text: str) -> List[str]:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text.strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [chunk for chunk in chunks if chunk]


def embed_texts(texts: List[str]) -> List[List[float]]:
    client = get_openai_client()
    embeddings = []
    for i in range(0, len(texts), 50):
        batch = texts[i : i + 50]
        response = client.embeddings.create(
            model=DEFAULT_EMBED_MODEL,
            input=batch,
            dimensions=EMBED_DIMENSIONS,
        )
        embeddings.extend([item.embedding for item in response.data])
        usage = getattr(response, "usage", None)
        if usage:
            print(f"[embeddings] tokens={usage.total_tokens}")
    return embeddings


def get_pinecone_index():
    client = Pinecone(api_key=PINECONE_API_KEY)
    return client.Index(name=PINECONE_INDEX_NAME, host=PINECONE_INDEX_HOST)


def validate_pinecone_index() -> Tuple[bool, Optional[str]]:
    try:
        index = get_pinecone_index()
        index.describe_index_stats()
        return True, None
    except Exception as exc:
        return False, str(exc)

def namespace_for_domain(domain: str) -> str:
    return slugify(domain)


def domain_index_exists(domain: str) -> bool:
    try:
        index = get_pinecone_index()
        namespace = namespace_for_domain(domain)
        stats = index.describe_index_stats()
        namespaces = stats.get("namespaces", {})
        if namespace not in namespaces:
            return False
        return namespaces[namespace].get("vector_count", 0) > 0
    except Exception as exc:
        _log(f"[pinecone] index lookup failed: {exc}")
        return False


def _log(message: str, log_fn: Optional[Callable[[str], None]] = None) -> None:
    if log_fn:
        log_fn(message)
    else:
        print(message)


def scrape_domain(domain: str, log_fn: Optional[Callable[[str], None]] = None) -> List[Dict[str, str]]:
    queries = generate_queries(domain)
    urls = []
    for query in queries:
        try:
            urls.extend(firecrawl_search(query, limit=3))
        except Exception as exc:
            _log(f"[search] failed for '{query}': {exc}", log_fn)

    urls = list(dict.fromkeys(urls))[:20]
    if not urls:
        raise RuntimeError("No URLs found for the domain.")

    _log(f"[scrape] found {len(urls)} URLs", log_fn)
    pages = []
    for url in urls:
        try:
            text, title, final_url = firecrawl_scrape(url)
            if text:
                pages.append({"text": text, "title": title, "url": final_url})
                _log(f"[scrape] ok: {final_url}", log_fn)
        except Exception as exc:
            _log(f"[scrape] failed: {url} ({exc})", log_fn)
    return pages


def index_domain(domain: str, log_fn: Optional[Callable[[str], None]] = None) -> None:
    if domain_index_exists(domain):
        _log("[index] domain already indexed in Pinecone. Removing old data before reindexing.", log_fn)
        index = get_pinecone_index()
        namespace = namespace_for_domain(domain)
        index.delete(delete_all=True, namespace=namespace)

    pages = scrape_domain(domain, log_fn=log_fn)
    index = get_pinecone_index()
    namespace = namespace_for_domain(domain)

    documents = []
    metadatas = []
    ids = []

    total_chunks = 0
    for page in pages:
        chunks = tokenize_chunks(page["text"])
        total_chunks += len(chunks)
        for chunk in chunks:
            documents.append(chunk)
            metadatas.append({"title": page["title"], "url": page["url"]})
            ids.append(str(uuid.uuid4()))

    if not documents:
        raise RuntimeError("No content collected to index.")

    embeddings = embed_texts(documents)
    vectors = []
    for idx, embedding in enumerate(embeddings):
        vectors.append(
            (
                ids[idx],
                embedding,
                {
                    "text": documents[idx],
                    "url": metadatas[idx]["url"],
                    "title": metadatas[idx]["title"],
                },
            )
        )

    index.upsert(vectors=vectors, namespace=namespace)
    _log(f"[index] stored {len(documents)} chunks from {len(pages)} pages", log_fn)
    _log(f"[index] average chunks per page: {max(1, total_chunks // max(1, len(pages)))}", log_fn)


def retrieve_context(domain: str, question: str) -> List[Dict[str, str]]:
    index = get_pinecone_index()
    namespace = namespace_for_domain(domain)
    client = get_openai_client()
    embedding = client.embeddings.create(
        model=DEFAULT_EMBED_MODEL,
        input=[question],
        dimensions=EMBED_DIMENSIONS,
    ).data[0].embedding

    results = index.query(
        vector=embedding,
        top_k=TOP_K,
        include_metadata=True,
        namespace=namespace,
    )

    contexts = []
    for match in results.get("matches", []):
        meta = match.get("metadata", {}) or {}
        text = meta.get("text", "")
        if text:
            contexts.append(
                {
                    "text": text,
                    "url": meta.get("url", ""),
                    "title": meta.get("title", ""),
                }
            )
    return contexts


def answer_question(domain: str, question: str) -> str:
    contexts = retrieve_context(domain, question)
    if not contexts:
        return "I don't know based on my knowledge base."
    context_block = "\n\n".join(
        [f"Source: {item['url']}\n{item['text']}" for item in contexts]
    )

    prompt = (
        f"Domain: {domain}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer based only on the context. If missing, say you don't know based on your knowledge base."
    )

    client = get_openai_client()
    response = client.chat.completions.create(
        model=DEFAULT_CHAT_MODEL,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    usage = getattr(response, "usage", None)
    if usage:
        print(f"[chat] tokens={usage.total_tokens}")

    return (response.choices[0].message.content or "").strip()


def chat_loop(domain: str) -> None:
    print("\nAsk questions about the domain. Type 'exit' to quit.\n")
    while True:
        question = input("You: ").strip()
        if question.lower() == "exit":
            break
        if not question:
            continue
        answer = answer_question(domain, question)
        print(f"\nAssistant: {answer}\n")


def run_streamlit() -> None:
    st.set_page_config(page_title="Domain Q&A Chatbot", layout="centered")
    st.title("Domain-Specific Q&A Chatbot")
    st.write("Select a domain, index sources, and ask questions from the scraped knowledge base.")

    if "logs" not in st.session_state:
        st.session_state.logs = []
    if "indexed_domain" not in st.session_state:
        st.session_state.indexed_domain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if not OPENAI_API_KEY or not FIRECRAWL_API_KEY:
        st.error("Missing required API keys in environment variables.")
        st.info("Required: OPENAI_API_KEY and FIRECRAWL_API_KEY")
        st.stop()

    domain = st.text_input("Domain", placeholder="e.g., Healthcare, Finance, AI, Law")
    index_col, status_col = st.columns([1, 2])

    log_box = st.empty()

    def log_fn(message: str) -> None:
        st.session_state.logs.append(message)
        log_box.code("\n".join(st.session_state.logs[-15:]))

    with index_col:
        index_clicked = st.button("Scrape & Index")

    with status_col:
        if domain:
            ok, error_msg = validate_pinecone_index()
            if not ok:
                st.error(f"Pinecone index error: {error_msg}")
            else:
                stats = get_pinecone_index().describe_index_stats()
                namespace = namespace_for_domain(domain)
                vector_count = stats.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
                st.info(f"Vectors in Pinecone for this domain: {vector_count}")
                if vector_count > 0:
                    st.success("Indexed data found. Click Scrape & Index to refresh.")
                else:
                    st.warning("No vectors found for this domain. Run Scrape & Index to ingest data.")

    if index_clicked:
        if not domain.strip():
            st.error("Please enter a domain before indexing.")
        else:
            st.session_state.logs = []
            with st.spinner("Indexing domain..."):
                try:
                    index_domain(domain.strip(), log_fn=log_fn)
                    st.session_state.indexed_domain = domain.strip()
                    st.success("Indexing completed.")
                    try:
                        index = get_pinecone_index()
                        stats = index.describe_index_stats()
                        namespace = namespace_for_domain(domain.strip())
                        vector_count = stats.get("namespaces", {}).get(namespace, {}).get("vector_count", 0)
                        st.info(f"Pinecone vectors stored for domain: {vector_count}")
                    except Exception as exc:
                        st.warning(f"Could not fetch Pinecone stats: {exc}")
                except Exception as exc:
                    st.error(f"Indexing failed: {exc}")

    if st.session_state.indexed_domain:
        st.subheader(f"Ask questions about {st.session_state.indexed_domain}")
        st.warning(
            "Answers are restricted to the indexed domain. If your question is out of scope, the assistant will say it does not know."
        )
        for role, content in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(content)

        user_question = st.chat_input("Ask a question")
        if user_question:
            st.session_state.chat_history.append(("user", user_question))
            with st.chat_message("user"):
                st.write(user_question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    answer = answer_question(st.session_state.indexed_domain, user_question)
                    st.write(answer)
            st.session_state.chat_history.append(("assistant", answer))


def run_cli() -> None:
    ensure_env()
    parser = argparse.ArgumentParser(description="Domain-Specific Q&A Chatbot")
    parser.add_argument("--domain", type=str, help="Domain name (e.g., Healthcare, AI)")
    args = parser.parse_args()

    domain = args.domain or input("Enter a domain: ").strip()
    if not domain:
        raise ValueError("Domain is required.")

    print(f"[domain] {domain}")
    print("[index] starting scrape and indexing...")
    index_domain(domain)
    chat_loop(domain)


if __name__ == "__main__":
    try:
        try:
            from streamlit.runtime import exists

            is_streamlit = exists()
        except Exception:
            is_streamlit = bool(getattr(st, "_is_running_with_streamlit", False))

        if is_streamlit:
            run_streamlit()
        else:
            run_cli()
    except KeyboardInterrupt:
        sys.exit(130)
