# LocallDeepseek_Agent (Local RAG)

A local chat app built on Ollama that uses `llama3.1:latest`, with optional RAG using Qdrant and optional web-search fallback using Exa.

API keys are not stored in this folder. The app reads configuration from the repo root `.env`.

## Features
- Local chat using `llama3.1:latest`
- Optional RAG mode (Qdrant) with PDF/URL ingestion
- Similarity threshold control for retrieval
- Optional web-search fallback (Exa)
- One-click indexing of the bundled `Resume__Anuj.pdf`

## Prerequisites
- Python 3.10+
- Ollama installed and running
- Qdrant (only if you want RAG mode)
- Exa API key (only if you want web-search fallback)

## Setup

```bash
cd /Users/anuj/Desktop/Anuj-AI-ML-Lab/All_LargeLangugage_Models/LocallDeepseek_Agent
python3 -m pip install -r requirements.txt
```

Pull required Ollama models:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

## Configuration (root `.env`)
File: `/Users/anuj/Desktop/Anuj-AI-ML-Lab/.env`

```bash
QDRANT_URL="https://your-qdrant.cloud:6333"
QDRANT_API_KEY="..."
OLLAMA_EMBED_MODEL="nomic-embed-text"
EXA_API_KEY="..."
```

Notes:
- If Qdrant is not configured, turn RAG mode off and use local chat only.
- If Exa is not configured, keep web-search fallback off.
- The app does not prompt for API keys in the UI.

## Run

```bash
streamlit run app.py
```


