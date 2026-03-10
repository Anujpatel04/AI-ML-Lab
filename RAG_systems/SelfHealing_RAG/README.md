# Self-Healing RAG

RAG pipeline that **evaluates retrieval quality** and **retries search** when relevance is low, then generates the final answer. Streamlit UI.

**Flow:** Query → Pinecone retrieval → LLM relevance check → retry retrieval if poor → answer generation (Azure OpenAI).

---

## Requirements

- Repo root `.env` with:
  - **Azure:** `AZURE_KEY`, `AZURE_ENDPOINT`, `API_VERSION` (optional: `AZURE_EMBEDDING_DEPLOYMENT`)
  - **Pinecone:** `PINECONE_API_KEY`; optional `PINECONE_INDEX` (default `memory-rag`), `PINECONE_NAMESPACE` (default `documents`), `PINECONE_HOST` (serverless index host if needed)
- Pinecone index with vectors in the chosen namespace; embedding dimension must match the Azure embedding model (e.g. 1536).

---

## Run

```bash
cd RAG_systems/SelfHealing_RAG
./run_app.sh
```

Or: `source ../../.venv/bin/activate && streamlit run app.py`

---

## UI

**Sidebar:** LLM and embedding model, top-k, relevance threshold, *Enable Self-Healing Retrieval* toggle.  
**Main:** Query → pipeline status (Retrieval → Relevance → Retry if triggered → Answer) → metrics → expanders for docs, relevance, retry log, and final answer.
