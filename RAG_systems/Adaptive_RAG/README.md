# Adaptive RAG

Adaptive Retrieval-Augmented Generation that selects retrieval strategy from query intent: **vector**, **graph** (Neo4j), or **hybrid**.

## Architecture

```
User Query → Query Classifier → factual | entity | broad
    → Vector Retriever (FAISS) | Graph Retriever (Neo4j) | Hybrid
    → Context → LLM (Azure OpenAI) → Answer
```

## Setup

1. Install dependencies (from this directory):

   ```bash
   pip install -r requirements.txt
   ```

2. Ensure `.env` at the **repository root** contains:

   - `AZURE_OPENAI_API_KEY` or `AZURE_KEY`
   - `AZURE_OPENAI_ENDPOINT` or `AZURE_ENDPOINT`
   - `AZURE_OPENAI_DEPLOYMENT` (or inferred from endpoint)
   - `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
   - `NEO4J_URI`, `NEO4J_USERNAME` or `NEO4J_USER`, `NEO4J_PASSWORD`

3. Ingest data (run from `Adaptive_RAG`):

   - **Vector:** place `.txt`/`.md` files in `data/documents/`, then:
     ```bash
     python ingestion/vector_ingest.py
     ```
   - **Graph:** same documents; entities and relations are extracted and written to Neo4j:
     ```bash
     python ingestion/graph_ingest.py
     ```

4. Start Neo4j (e.g. Docker) if using graph or hybrid retrieval.

## Run

**CLI** (from the `Adaptive_RAG` directory):

```bash
python main.py
```

**Streamlit UI** (use the script so the repo `.venv` is used and faiss/neo4j are available):

```bash
./run_app.sh
```

Or from repo root with the venv activated: `source .venv/bin/activate` then `cd RAG_systems/Adaptive_RAG && streamlit run app.py`.

Enter a question; the system classifies it and uses vector, graph, or hybrid retrieval before generating the answer. The app shows the answer and an expandable "Retrieval" section with the strategy and context used.

## Project layout

- `main.py` – CLI entry point  
- `app.py` – Streamlit frontend  
- `router.py` – classification and routing to retrievers  
- `config.py` – env-based configuration (loads repo root `.env`)  
- `classifiers/query_classifier.py` – LLM-based query type (factual / entity / broad)  
- `retrievers/` – vector (FAISS), graph (Neo4j), hybrid  
- `llm/generator.py` – Azure OpenAI answer generation  
- `ingestion/` – vector and graph ingestion scripts  
- `utils/logger.py` – logging

