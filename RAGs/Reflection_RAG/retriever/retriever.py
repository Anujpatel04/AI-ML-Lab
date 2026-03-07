"""
Retriever: load documents, chunk, embed with OpenAI, store in FAISS or numpy fallback, return top-k.
"""

import logging
import pickle
from pathlib import Path
from typing import List, Optional, Any

from utils.document_loader import load_documents, chunk_text

logger = logging.getLogger(__name__)


def _faiss_available() -> bool:
    try:
        import faiss  # noqa: F401
        return True
    except ImportError:
        return False


class Retriever:
    """
    Build vector index from data/ and query for top-k relevant chunks.
    Uses FAISS when available; falls back to numpy L2 search (e.g. Python 3.14).
    Persists index and chunk list to disk for reuse.
    Works with OpenAI or Azure OpenAI client.
    """

    def __init__(
        self,
        data_dir: Path,
        index_path: Path,
        chunks_path: Path,
        client,  # OpenAI or AzureOpenAI
        embedding_model: str = "text-embedding-3-small",
        chunk_size: int = 800,
        chunk_overlap: int = 100,
    ):
        self.data_dir = Path(data_dir)
        self.index_path = Path(index_path)
        self.chunks_path = Path(chunks_path)
        self.embeddings_path = self.index_path.parent / "embeddings.npy"
        self.client = client
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._chunks: List[str] = []
        self._index: Optional[Any] = None
        self._use_faiss = _faiss_available()

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Call OpenAI embeddings API; batch if needed."""
        out = []
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            resp = self.client.embeddings.create(input=batch, model=self.embedding_model)
            for item in resp.data:
                out.append(item.embedding)
        return out

    def build_index(self) -> None:
        """Load documents, chunk, embed, build index (FAISS or numpy), and save to disk."""
        import numpy as np

        docs = load_documents(self.data_dir)
        if not docs:
            raise ValueError(f"No documents found in {self.data_dir}. Add .txt or .md files.")

        self._chunks = chunk_text(docs, self.chunk_size, self.chunk_overlap)
        if not self._chunks:
            raise ValueError("No chunks produced from documents.")

        logger.info("Embedding %s chunks...", len(self._chunks))
        embeddings = self._get_embeddings(self._chunks)
        matrix = np.array(embeddings, dtype=np.float32)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        if self._use_faiss:
            import faiss
            index = faiss.IndexFlatL2(matrix.shape[1])
            index.add(matrix)
            faiss.write_index(index, str(self.index_path))
            self._index = index
            logger.info("Index saved (FAISS) to %s", self.index_path)
        else:
            np.save(str(self.embeddings_path), matrix)
            self._index = matrix
            logger.info("Index saved (numpy) to %s", self.embeddings_path)

        with open(self.chunks_path, "wb") as f:
            pickle.dump(self._chunks, f)

    def load_index(self) -> bool:
        """Load existing index and chunks from disk. Returns True if loaded."""
        if not self.chunks_path.is_file():
            return False
        if self._use_faiss and self.index_path.is_file():
            import faiss
            self._index = faiss.read_index(str(self.index_path))
            with open(self.chunks_path, "rb") as f:
                self._chunks = pickle.load(f)
            logger.info("Loaded FAISS index with %s chunks", len(self._chunks))
            return True
        if self.embeddings_path.is_file():
            import numpy as np
            self._index = np.load(str(self.embeddings_path))
            with open(self.chunks_path, "rb") as f:
                self._chunks = pickle.load(f)
            logger.info("Loaded numpy index with %s chunks", len(self._chunks))
            return True
        return False

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        """
        Embed query, search index, return list of top-k chunk texts.
        Builds index from data/ if index does not exist.
        """
        if self._index is None and not self.load_index():
            self.build_index()

        import numpy as np
        query_embedding = np.array([self._get_embeddings([query])[0]], dtype=np.float32)
        k = min(top_k, len(self._chunks))

        if self._use_faiss:
            scores, indices = self._index.search(query_embedding, k)
            indices = indices[0]
        else:
            matrix = self._index
            distances = np.sum((matrix - query_embedding) ** 2, axis=1)
            indices = np.argsort(distances)[:k]

        return [self._chunks[i] for i in indices if 0 <= i < len(self._chunks)]
