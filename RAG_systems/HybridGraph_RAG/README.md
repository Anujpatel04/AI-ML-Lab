# Hybrid Graph + Vector RAG

Graph and vector hybrid RAG: **entity extraction** → **Neo4j graph search** → **Pinecone vector retrieval** → **context fusion** → **LLM answer**. Uses **repo root** `.env` and `.venv` only.

## Pipeline

1. **Entity extraction** — LLM extracts entities (people, orgs, products, topics) from the query.
2. **Graph search** — Neo4j returns relationships involving those entities.
3. **Vector retrieval** — Pinecone returns top-k semantic document chunks.
4. **Context fusion** — Graph + documents merged into one context block.
5. **LLM** — Answer generated from fused context.

## Setup

1. **Env (repo root `.env` only)**  
   Set:
   - `NEO4J_URI`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
   - `PINECONE_API_KEY`, `PINECONE_INDEX` (default `hybrid-graph-rag`)
   - `OPENAI_API_KEY` or Azure (`AZURE_ENDPOINT`, `AZURE_KEY`, `API_VERSION`)

2. **Neo4j**  
   Run Neo4j (local or cloud). Optional: create nodes with `name` or `title` and relationships; the pipeline will query them. If the graph is empty, graph context is omitted.  
   **Docker (local):** from repo root run  
   `./RAGs/HybridGraph_RAG/run_neo4j.sh`  
   (uses `NEO4J_PASSWORD` from `.env`; browser UI http://localhost:7474, bolt localhost:7687). Stop with `docker stop neo4j`.

3. **Pinecone**  
   The **CLI** can create a serverless index (AWS `us-east-1`, dimension 1536) and index docs on first run. The **Streamlit app** does not create the index or index docs (to avoid hanging). First-time setup: run once from CLI to create index and index `data/`:  
   `cd RAGs/HybridGraph_RAG && python -c "from rag_pipeline import run; run('test', index_docs_if_empty=True)"`  
   Add `.txt`/`.md` under `data/` for vector indexing.

4. **Install (repo root `.venv`)**  
   ```bash
   source .venv/bin/activate
   pip install -r RAGs/HybridGraph_RAG/requirements.txt
   ```

## Run

From repo root with `.venv` activated:

```bash
source .venv/bin/activate
cd RAGs/HybridGraph_RAG
streamlit run app.py
```

Or: `./RAGs/HybridGraph_RAG/run_app.sh`

CLI: `python rag_pipeline.py "Your question"`

**If indexing in the app times out:** run indexing from the terminal (from repo root):  
`cd RAGs/HybridGraph_RAG && python index_docs.py`  
Then use the app only for questions.

**If the app hangs or errors:** open **Test connections** in the app and click **Run connection check**. Or:  
`python -m RAGs.HybridGraph_RAG.check_connections`

## Project layout

- `config.py` — Loads repo root `.env`; Neo4j, Pinecone, OpenAI/Azure.
- `entity_extractor.py` — LLM-based entity extraction from the query.
- `graph_store.py` — Neo4j connection and Cypher query for entities/relationships.
- `vector_store.py` — Pinecone embed + index/retrieve document chunks.
- `hybrid_retriever.py` — Combines graph + vector into one context string.
- `rag_pipeline.py` — Orchestrates full pipeline and LLM answer.
- `utils/document_loader.py` — Load and chunk `.txt`/`.md` from `data/`.
- `app.py` — Streamlit UI.

Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab).
