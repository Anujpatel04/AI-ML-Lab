# Multi-Agent RAG

Multi-agent RAG system where a **Retriever**, **Reasoning**, and **Verification** agent collaborate to answer questions using retrieved knowledge. Includes a Streamlit UI.

## Architecture

```
User Question (Streamlit) → Retriever Agent (FAISS + SentenceTransformers)
  → Reasoning Agent (Azure OpenAI) → Verification Agent (grounding check / refinement)
  → Final answer + sources
```

## Setup

1. **Install dependencies** (from repo root, using `.venv` recommended):

   ```bash
   pip install -r RAG_systems/MultiAgent_RAG/requirements.txt
   ```

2. **Environment**: Load from repo root `.env`. Required:

   - `AZURE_OPENAI_API_KEY` or `AZURE_KEY`
   - `AZURE_OPENAI_ENDPOINT` or `AZURE_ENDPOINT`
   - `API_VERSION` (optional)

3. **Documents**: Add `.txt` or `.md` files under `data/documents/`. The FAISS index is built automatically on first query if missing.

## Run

**Streamlit UI** (recommended: use script so repo `.venv` is used):

```bash
cd RAG_systems/MultiAgent_RAG
./run_app.sh
```

Or with venv activated: `streamlit run frontend/app.py`

**CLI**:

```bash
cd RAG_systems/MultiAgent_RAG
python main.py
```

## Project structure

- `agents/` – Retriever, Reasoning, Verifier agents
- `core/` – Orchestrator, vector store (FAISS + SentenceTransformers), prompts
- `utils/` – config (loads repo `.env`), logger
- `frontend/app.py` – Streamlit UI
- `data/documents/` – source documents (index built on first run if needed)

## Requirements

- Python 3.10+
- Azure OpenAI for reasoning and verification
- SentenceTransformers for embeddings (local)
- FAISS for vector search
