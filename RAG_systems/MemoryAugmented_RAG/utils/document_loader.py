"""
Load documents from data/ and split into chunks for indexing.
Supports .txt and .md; optional PDF if pypdf installed.
"""

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def load_documents(data_dir: Path) -> List[str]:
    """
    Load all text from supported files in data_dir.
    Returns a list of document contents (one string per file).
    """
    if not data_dir.is_dir():
        logger.warning("Data dir %s not found or not a directory", data_dir)
        return []

    texts: List[str] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        suf = path.suffix.lower()
        if suf == ".txt":
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                texts.append(content)
                logger.info("Loaded %s (%s chars)", path.name, len(content))
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)
        elif suf == ".md":
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                texts.append(content)
                logger.info("Loaded %s (%s chars)", path.name, len(content))
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)
        elif suf == ".pdf":
            try:
                content = _read_pdf(path)
                if content:
                    texts.append(content)
                    logger.info("Loaded %s (%s chars)", path.name, len(content))
            except Exception as e:
                logger.warning("Failed to load PDF %s: %s", path, e)
    return texts


def _read_pdf(path: Path) -> str:
    """Extract text from PDF if pypdf is available."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    except ImportError:
        logger.warning("pypdf not installed; skipping PDF %s", path)
        return ""


def chunk_text(
    documents: List[str],
    chunk_size: int = 800,
    overlap: int = 100,
) -> List[str]:
    """Split documents into overlapping chunks for embedding."""
    if overlap >= chunk_size:
        overlap = chunk_size // 2
    chunks: List[str] = []
    for doc in documents:
        chunks.extend(_split_into_chunks(doc, chunk_size, overlap))
    return chunks


def _split_into_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Split a single document into chunks with overlap."""
    if not text.strip():
        return []
    out = []
    start = 0
    text = text.replace("\r\n", "\n")
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunk = text[start:].strip()
            if chunk:
                out.append(chunk)
            break
        break_at = text.rfind("\n", start, end + 1)
        if break_at > start:
            end = break_at + 1
        else:
            break_at = text.rfind(" ", start, end + 1)
            if break_at > start:
                end = break_at + 1
        chunk = text[start:end].strip()
        if chunk:
            out.append(chunk)
        start = end - overlap
        if start <= 0:
            start = end
    return out
