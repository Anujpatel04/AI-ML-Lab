# Memory-Augmented RAG

Long-term conversational memory plus document retrieval using **Pinecone** and **OpenAI**. Pipeline: memory search → document retrieval → LLM answer → store new memory.

## Pipeline

1. **Memory search** — Retrieve relevant past conversations from Pinecone (namespace `memory`).
2. **Document retrieval** — Vector search over the knowledge base in Pinecone (namespace `documents`).
3. **LLM answer** — Generate response using memories + documents.
4. **Store new memory** — Save the Q&A turn in Pinecone for future queries.

## Setup

1. **Env** — In the **repo root** `.env` set:
   - `PINECONE_API_KEY` (required)
   - `PINECONE_INDEX` (default: `memory-rag`)
   - `PINECONE_ENVIRONMENT` (optional)
   - `OPENAI_API_KEY` (for embeddings and chat)

2. **Pinecone index** — Create an index in the Pinecone console (or with the SDK) with **dimension 1536** (OpenAI `text-embedding-3-small`). Use the same name as `PINECONE_INDEX`.

3. **Install** (from this directory):
   ```bash
   pip install -r requirements.txt
   ```

4. **Documents** — Add `.txt` or `.md` files under `data/`. On first run, the pipeline will index them into the `documents` namespace.

## Run

From `RAGs/MemoryAugmented_RAG`:

```bash
python rag_pipeline.py "Your question here"
```

Or pipe a question:

```bash
echo "What did we discuss about X?" | python rag_pipeline.py
```

## Project layout

- `config.py` — Loads repo root `.env`; Pinecone and OpenAI settings.
- `memory_store.py` — `store_memory(text)`, `search_memory(query)` (Pinecone namespace `memory`).
- `retriever.py` — Document load/chunk/embed, upsert to Pinecone namespace `documents`; `retrieve_documents(query)`.
- `rag_pipeline.py` — Orchestrates: memory → documents → LLM → store memory; `run(query)` and CLI.
- `utils/document_loader.py` — Load and chunk `.txt`/`.md` from `data/`.
- `data/` — Knowledge base files (indexed into Pinecone on first use).

Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab).
