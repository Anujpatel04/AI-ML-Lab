"""
Load internal documents (.md, .txt, .pdf) from a directory for RAG ingestion.
"""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils.config import CHUNK_OVERLAP, CHUNK_SIZE
from utils.logger import get_logger

logger = get_logger(__name__)

# Unstructured file types we support via DirectoryLoader glob + loader_cls
TEXT_EXTENSIONS = {".md", ".txt"}


def _load_text_files(directory: Path) -> list[Document]:
    docs: list[Document] = []
    for ext in TEXT_EXTENSIONS:
        loader = DirectoryLoader(
            str(directory),
            glob=f"**/*{ext}",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"},
        )
        try:
            docs.extend(loader.load())
        except Exception as e:
            logger.warning("document_loader | text_glob_failed | ext=%s | error=%s", ext, e)
    return docs


def _load_pdf_files(directory: Path) -> list[Document]:
    docs: list[Document] = []
    for pdf in directory.rglob("*.pdf"):
        try:
            docs.extend(PyPDFLoader(str(pdf)).load())
        except Exception as e:
            logger.warning("document_loader | pdf_failed | path=%s | error=%s", pdf, e)
    return docs


def load_documents_from_directory(directory: Path) -> list[Document]:
    """
    Load all supported documents under ``directory`` (recursive).
    """
    if not directory.is_dir():
        raise FileNotFoundError(f"Knowledge base path is not a directory: {directory}")
    all_docs = _load_text_files(directory) + _load_pdf_files(directory)
    logger.info("document_loader | loaded_raw | count=%s | path=%s", len(all_docs), directory)
    return all_docs


def split_documents(documents: list[Document]) -> list[Document]:
    """Chunk documents for embedding."""
    if not documents:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
    )
    chunks = splitter.split_documents(documents)
    logger.info("document_loader | chunks | count=%s", len(chunks))
    return chunks
