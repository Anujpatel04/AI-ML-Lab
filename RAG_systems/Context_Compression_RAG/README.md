# Context Compression RAG

RAG pipeline that retrieves 20 document chunks, compresses them with an LLM to 5 sections, then generates the answer. Reduces context size before the final LLM call. Uses Azure OpenAI and FAISS.

**Pipeline:** Query → Embed → Retrieve (top 20) → Compress (LLM → top 5) → Answer.

---

## Setup

**Environment:** Load Azure settings from repo root `.env`.

 `AZURE_OPENAI_API_KEY` or `AZURE_KEY`  API key 
 `AZURE_OPENAI_ENDPOINT` or `AZURE_ENDPOINT`  Endpoint URL (deployment parsed if needed) 
 `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` or `AZURE_EMBEDDING_DEPLOYMENT`  Embedding model (default: `text-embedding-ada-002`) 
 `AZURE_OPENAI_API_VERSION` or `API_VERSION`  API version 

**Install (repo root):**

```bash
.venv/bin/pip install -r RAGs/Context_Compression_RAG/requirements.txt
```

---

## Run

From repo root:

```bash
.venv/bin/python -m streamlit run RAGs/Context_Compression_RAG/app.py
```

From project directory (uses repo `.venv`):

```bash
cd RAGs/Context_Compression_RAG
./run_app.sh
```

---

## Usage

1. Upload PDF(s).
2. Wait for indexing (chunking, embeddings, FAISS).
3. Enter a query and click **Get answer**.
4. View retrieved chunks (20), compressed context (5), and final answer.

---

## Requirements

- Python 3.10+
- Repo root `.env` with Azure OpenAI credentials
- `streamlit`, `openai`, `pypdf`, `python-dotenv`, `faiss-cpu`, `numpy` (see `requirements.txt`)

Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab).
