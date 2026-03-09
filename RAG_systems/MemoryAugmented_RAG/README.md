# Memory-Augmented RAG

Long-term conversational memory plus document retrieval using **Pinecone** and **OpenAI**. Pipeline: memory search → document retrieval → LLM answer → store new memory.

## Pipeline

1. **Memory search** — Retrieve relevant past conversations from Pinecone (namespace `memory`).
2. **Document retrieval** — Vector search over the knowledge base in Pinecone (namespace `documents`).
3. **LLM answer** — Generate response using memories + documents.
4. **Store new memory** — Save the Q&A turn in Pinecone for future queries.

## Setup

1. **Env** — Use the **repo root** `.env` only (no project-local .env). Set:
   - `PINECONE_API_KEY` (required)
   - `PINECONE_INDEX` (default: `memory-rag`)
   - `PINECONE_ENVIRONMENT` (optional)
   - **OpenAI:** `OPENAI_API_KEY` (for embeddings and chat), or
   - **Azure OpenAI:** `AZURE_ENDPOINT`, `AZURE_KEY`, `API_VERSION` (optional `AZURE_EMBEDDING_DEPLOYMENT` for embeddings)

2. **Pinecone index** — The app auto-creates a serverless index (AWS `us-east-1`, dimension 1536, cosine) if missing. If you already created an index in another region (e.g. GCP), delete it in the Pinecone console (or `pc.delete_index("memory-rag")`) and run again so it is recreated in AWS `us-east-1`.

3. **Install** (from this directory):
   ```bash
   pip install -r requirements.txt
   ```

4. **Documents** — Add `.txt` or `.md` files under `data/`. On first run, the pipeline will index them into the `documents` namespace.

## Run

Uses **repo root** `.venv` and **repo root** `.env` only (no project-local env).

**Streamlit UI (recommended):** From repo root, activate `.venv` then run:

```bash
source .venv/bin/activate
cd RAGs/MemoryAugmented_RAG
streamlit run app.py
```

Or from anywhere (uses repo .venv): `./RAGs/MemoryAugmented_RAG/run_app.sh`

**CLI:** From `RAGs/MemoryAugmented_RAG`:

```bash
python rag_pipeline.py "Your question here"
```

Or pipe a question: `echo "What did we discuss about X?" | python rag_pipeline.py`

## Project layout

- `config.py` — Loads repo root `.env`; Pinecone and OpenAI settings.
- `memory_store.py` — `store_memory(text)`, `search_memory(query)` (Pinecone namespace `memory`).
- `retriever.py` — Document load/chunk/embed, upsert to Pinecone namespace `documents`; `retrieve_documents(query)`.
- `rag_pipeline.py` — Orchestrates: memory → documents → LLM → store memory; `run(query)` and CLI.
- `utils/document_loader.py` — Load and chunk `.txt`/`.md` from `data/`.
- `data/` — Knowledge base files (indexed into Pinecone on first use).

Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab).
