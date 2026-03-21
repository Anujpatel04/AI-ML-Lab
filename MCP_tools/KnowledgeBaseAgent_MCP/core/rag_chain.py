"""
RAG: retrieve relevant chunks and generate an answer with the configured LLM.

Uses LangChain Core only (no ``langchain.chains``) for compatibility with slim installs.
"""

from langchain_core.prompts import ChatPromptTemplate

from core.vector_store import get_vectorstore
from utils.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_BASE_URL,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
    RETRIEVER_K,
)
from utils.logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """You are an internal knowledge base assistant for employees and support teams.
Answer using ONLY the provided context from company documents.
If the context does not contain enough information, say clearly that the knowledge base does not document this and suggest what kind of document might be needed.
Be concise and professional."""


def _get_llm():
    if LLM_PROVIDER == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_BASE_URL,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    )


def _retrieve_documents(retriever, query: str):
    """Support both ``invoke`` and legacy ``get_relevant_documents``."""
    if hasattr(retriever, "invoke"):
        return retriever.invoke(query)
    return retriever.get_relevant_documents(query)


def _format_context(docs) -> str:
    return "\n\n".join((d.page_content or "").strip() for d in docs if d.page_content)


def answer_question(question: str) -> dict:
    """
    Run RAG and return answer plus source file hints.

    Returns:
        dict with keys: answer (str), sources (list of dict with source, snippet)
    """
    if not question.strip():
        raise ValueError("Question must not be empty")

    from core.vector_store import collection_count

    if collection_count() == 0:
        raise RuntimeError(
            "Knowledge base is empty. POST /ingest with a docs path first, "
            "or add files under kb_documents/ and ingest."
        )

    retriever = get_vectorstore().as_retriever(search_kwargs={"k": RETRIEVER_K})
    q = question.strip()
    logger.info("rag_chain | query | length=%s", len(q))

    context_docs = _retrieve_documents(retriever, q)
    context = _format_context(context_docs)

    llm = _get_llm()
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Context from internal documents:\n{context}\n\nQuestion: {input}",
            ),
        ]
    )
    messages = prompt.format_messages(context=context, input=q)
    response = llm.invoke(messages)
    answer = (getattr(response, "content", None) or str(response)).strip()

    sources: list[dict] = []
    seen: set[str] = set()
    for doc in context_docs:
        src = (doc.metadata or {}).get("source") or "unknown"
        if src not in seen:
            seen.add(src)
            body = doc.page_content or ""
            snippet = body[:200].replace("\n", " ")
            sources.append(
                {
                    "source": str(src),
                    "snippet": snippet + ("..." if len(body) > 200 else ""),
                }
            )

    return {"answer": answer, "sources": sources}
