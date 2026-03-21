"""
Chroma vector store + embedding factory for the knowledge base.
"""

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from utils.config import (
    AZURE_EMBEDDING_DEPLOYMENT,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_BASE_URL,
    CHROMA_PERSIST_DIR,
    KB_EMBEDDING_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OPENAI_API_KEY,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def get_embeddings() -> Embeddings:
    """Embedding model: Azure (default), OpenAI API, Ollama, or local HuggingFace."""
    if KB_EMBEDDING_PROVIDER == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        logger.info("vector_store | embeddings=huggingface")
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if KB_EMBEDDING_PROVIDER == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings
        logger.info("vector_store | embeddings=ollama")
        return OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_EMBED_MODEL)
    if KB_EMBEDDING_PROVIDER == "openai" and OPENAI_API_KEY:
        from langchain_openai import OpenAIEmbeddings
        logger.info("vector_store | embeddings=openai")
        return OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    # Default: Azure OpenAI embeddings
    from langchain_openai import AzureOpenAIEmbeddings
    if not AZURE_OPENAI_API_KEY or not AZURE_OPENAI_BASE_URL:
        raise RuntimeError(
            "Azure embeddings require AZURE_KEY and AZURE_ENDPOINT in .env, "
            "or set KB_EMBEDDING_PROVIDER=huggingface for local embeddings."
        )
    logger.info("vector_store | embeddings=azure | deployment=%s", AZURE_EMBEDDING_DEPLOYMENT)
    return AzureOpenAIEmbeddings(
        azure_endpoint=AZURE_OPENAI_BASE_URL,
        api_key=AZURE_OPENAI_API_KEY,
        openai_api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_EMBEDDING_DEPLOYMENT,
    )


def get_vectorstore() -> Chroma:
    """Load or create persistent Chroma collection."""
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    emb = get_embeddings()
    return Chroma(
        persist_directory=str(CHROMA_PERSIST_DIR),
        embedding_function=emb,
        collection_name="knowledge_base",
    )


def ingest_documents(chunks: list[Document], reset: bool = False) -> int:
    """
    Add document chunks to Chroma. If ``reset``, wipe the index first.
    If the index already has vectors and ``reset`` is false, new chunks are appended.
    """
    import shutil

    if not chunks:
        return 0
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    emb = get_embeddings()
    if reset:
        if CHROMA_PERSIST_DIR.exists():
            shutil.rmtree(CHROMA_PERSIST_DIR)
        CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("vector_store | reset_chroma | path=%s", CHROMA_PERSIST_DIR)

    existing = collection_count()
    if reset or existing == 0:
        Chroma.from_documents(
            documents=chunks,
            embedding=emb,
            persist_directory=str(CHROMA_PERSIST_DIR),
            collection_name="knowledge_base",
        )
    else:
        vs = Chroma(
            persist_directory=str(CHROMA_PERSIST_DIR),
            embedding_function=emb,
            collection_name="knowledge_base",
        )
        vs.add_documents(chunks)
    logger.info("vector_store | ingested | chunks=%s", len(chunks))
    return len(chunks)


def collection_count() -> int:
    """Approximate number of vectors (0 if store missing)."""
    if not CHROMA_PERSIST_DIR.is_dir():
        return 0
    try:
        vs = get_vectorstore()
        return vs._collection.count()  # type: ignore[attr-defined]
    except Exception:
        return 0
