"""Load documents from data/ and split into chunks. Supports .txt, .md; optional .pdf."""

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def load_documents(data_dir: Path) -> List[str]:
    """Load all text from supported files in data_dir. Returns list of document contents."""
    if not data_dir.is_dir():
        logger.warning("Data dir %s not found or not a directory", data_dir)
        return []
    texts: List[str] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        suf = path.suffix.lower()
        if suf in (".txt", ".md"):
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                texts.append(content)
            except Exception as e:
                logger.warning("Failed to load %s: %s", path, e)
        elif suf == ".pdf":
            try:
                from pypdf import PdfReader
                reader = PdfReader(path)
                content = "\n".join(p.extract_text() or "" for p in reader.pages)
                if content.strip():
                    texts.append(content)
            except Exception as e:
                logger.warning("Failed to load PDF %s: %s", path, e)
    return texts


def chunk_text(documents: List[str], chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Split documents into overlapping chunks."""
    if overlap >= chunk_size:
        overlap = chunk_size // 2
    out: List[str] = []
    for doc in documents:
        doc = doc.replace("\r\n", "\n").strip()
        if not doc:
            continue
        start = 0
        while start < len(doc):
            end = min(start + chunk_size, len(doc))
            if end < len(doc):
                br = doc.rfind("\n", start, end + 1)
                if br > start:
                    end = br + 1
                else:
                    br = doc.rfind(" ", start, end + 1)
                    if br > start:
                        end = br + 1
            chunk = doc[start:end].strip()
            if chunk:
                out.append(chunk)
            if end >= len(doc):
                break
            start = end - overlap
            if start <= 0:
                start = end
    return out
