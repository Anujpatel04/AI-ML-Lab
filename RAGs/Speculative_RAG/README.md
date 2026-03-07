# Speculative RAG

Production-style Speculative Retrieval-Augmented Generation: a small model produces a fast draft answer, and a larger model verifies and improves it using retrieved context.

## Pipeline

1. **Retrieve** — Top-k relevant chunks from the vector index (FAISS or numpy).
2. **Draft** — Small model (e.g. gpt-4o-mini) generates an initial answer from the question only.
3. **Verify** — Large model (e.g. gpt-4o) checks the draft against retrieved context, corrects hallucinations, and returns a final answer.

## Setup

1. Install dependencies (from this directory):

   ```bash
   pip install -r requirements.txt
   ```

2. Configure API keys in the **repo root** `.env` (see `.env.example`):
   - **OpenAI:** `OPENAI_API_KEY=sk-...`
   - **Azure OpenAI:** `AZURE_ENDPOINT`, `AZURE_KEY`, `API_VERSION` (optional draft/verifier deployments: `AZURE_DRAFT_DEPLOYMENT`, `AZURE_VERIFIER_DEPLOYMENT`)

3. Put `.txt` or `.md` (and optionally `.pdf`) files in `data/` for indexing.

## Run

```bash
streamlit run app.py
```

From the project root: `RAGs/Speculative_RAG`, so:

```bash
cd RAGs/Speculative_RAG
streamlit run app.py
```

## Project layout

- `app.py` — Streamlit UI
- `config.py` — Loads repo root `.env`; OpenAI vs Azure; model and path settings
- `retriever/` — Document loading, chunking, embeddings, FAISS/numpy index, top-k retrieval
- `generator/` — Draft answer from the small model
- `verifier/` — Verify and improve draft using the large model and context
- `pipeline/` — Orchestrates retrieve → draft → verify
- `utils/` — Document loader
- `data/` — Source documents (index built on first run or when index is missing)

Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab).
