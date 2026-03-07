# Reflection RAG

Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab/tree/main).

Retrieval-Augmented Generation with self-critique: retrieve docs, generate an initial answer, then use the LLM to critique and produce an improved final answer.

**Pipeline:** User question -> Retrieve (FAISS) -> Initial answer (OpenAI) -> Reflection (critique + improved answer) -> Final answer.

## Requirements

- Python 3.9+
- **Repo root** `.env` (parent of `RAGs`) with either:
  - **OpenAI:** `OPENAI_API_KEY`, or

## Installation

```bash
cd RAGs/Reflection_RAG
pip install -r requirements.txt
```

## Data

Put source documents (`.txt`, `.md`, or `.pdf`) in `data/`. On first query, the app builds a FAISS index and reuses it until you add or change files (delete the `index/` folder to force rebuild).

## Run

```bash
streamlit run app.py
```

## Layout

- **config.py** – Paths, chunk/retrieval/API settings; loads `.env` from repo root.
- **utils/document_loader.py** – Load and chunk documents from `data/`.
- **retriever/retriever.py** – Embed with OpenAI, store in FAISS, return top-k chunks.
- **generator/answer_generator.py** – Initial answer from context via OpenAI Chat.
- **reflection/reflection_agent.py** – Critique initial answer and return improved answer.
- **pipeline/rag_pipeline.py** – Runs retrieve -> generate -> reflect and returns all outputs.

## Optional env (repo root `.env`)

- `RAG_TOP_K` (default 5)
- `RAG_CHUNK_SIZE` (800), `RAG_CHUNK_OVERLAP` (100)
- `RAG_EMBEDDING_MODEL` (text-embedding-3-small), `RAG_CHAT_MODEL` (gpt-4o-mini)
