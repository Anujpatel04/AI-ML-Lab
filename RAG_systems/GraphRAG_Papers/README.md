# Graph-RAG arXiv

A production-quality Graph-Based Retrieval Augmented Generation (Graph-RAG) system built on arXiv metadata and Neo4j. The system prioritizes graph-first reasoning over flat vector search and is designed for public hosting and technical interviews.

## Architecture Overview

- **Ingestion**: Fetches arXiv XML, parses and normalizes fields, maps arXiv categories into readable topics, and ingests nodes/relationships into Neo4j with idempotent Cypher.
- **Retrieval**: Extracts entities and intent from user queries, performs deterministic graph traversals, and gathers paper abstracts.
- **LLM Synthesis**: Summarizes retrieved evidence and generates answers without hallucination.
- **Frontend**: Minimal, professional UI built with Next.js and Tailwind CSS.

## Graph Schema

Nodes:
- `Paper {id, title, year, abstract}`
- `Author {name}`
- `Topic {name}`

Relationships:
- `(Paper)-[:WRITTEN_BY]->(Author)`
- `(Paper)-[:HAS_TOPIC]->(Topic)`

Constraints:
- Unique `Paper.id`, `Author.name`, `Topic.name`

Schema file: [backend/neo4j/schema.cypher](backend/neo4j/schema.cypher)

## Setup

### 1) Start Neo4j

Use Docker:

- Run `docker-compose up -d neo4j`
- Neo4j browser: http://localhost:7474

### 2) Backend

- Create a virtual environment
- Install dependencies from [backend/requirements.txt](backend/requirements.txt)
- Copy [backend/.env.example](backend/.env.example) to `backend/.env` and set values
- Run the API: `uvicorn app.main:app --reload --port 8000`

### 3) Ingest arXiv data

- Run: `python -m app.ingest.arxiv_ingest`
- Default query: "graph neural networks" (override via `ARXIV_QUERY`)

### 4) Frontend

- Install dependencies in the frontend folder
- Run `npm run dev` and open http://localhost:3000

## API Endpoints

- `POST /query` – graph-first retrieval + LLM synthesis
- `GET /graph/stats` – counts of nodes and relationships
- `GET /health` – service status

## Example Queries

- “Which authors publish most on graph neural networks?”
- “Show recent papers on graph representation learning.”
- “What topics are associated with knowledge graphs in this dataset?”

### Example Output (structure)

```
Answer:
  - Summary derived strictly from retrieved abstracts.

Referenced Papers:
  - Title, Authors, Year
  - Title, Authors, Year
```

If no relevant data is found, the system responds with:

```
Insufficient information in the current knowledge graph.
```

## Engineering Notes

- Graph ingestion is idempotent via `MERGE` and uniqueness constraints.
- Retrieval always uses graph traversals; no vector search is required.
- LLM responses are constrained to retrieved context only.

## Project Structure

- [backend/app/main.py](backend/app/main.py) – FastAPI app
- [backend/app/graph.py](backend/app/graph.py) – Neo4j client & Cypher operations
- [backend/app/retrieval.py](backend/app/retrieval.py) – entity extraction & graph retrieval
- [backend/app/ingest/arxiv_ingest.py](backend/app/ingest/arxiv_ingest.py) – arXiv ingestion
- [frontend/app/page.tsx](frontend/app/page.tsx) – UI

## Docker

Full stack:

- `docker-compose up --build`

This starts Neo4j, FastAPI, and the Next.js frontend.
