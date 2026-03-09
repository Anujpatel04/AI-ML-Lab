from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .logging import configure_logging
from .graph import Neo4jClient
from .retrieval import retrieve_papers
from .llm import synthesize_answer
from .models import QueryRequest, QueryResponse, GraphStats, Paper


configure_logging()
app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    app.state.neo4j = Neo4jClient()


@app.on_event("shutdown")
def _shutdown() -> None:
    app.state.neo4j.close()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/graph/stats", response_model=GraphStats)
def graph_stats() -> GraphStats:
    client: Neo4jClient = app.state.neo4j
    result = client.run(
        """
        MATCH (p:Paper) WITH count(p) AS paper_count
        MATCH (a:Author) WITH paper_count, count(a) AS author_count
        MATCH (t:Topic) WITH paper_count, author_count, count(t) AS topic_count
        MATCH ()-[r:WRITTEN_BY]->() WITH paper_count, author_count, topic_count, count(r) AS written_by_count
        MATCH ()-[r:HAS_TOPIC]->() WITH paper_count, author_count, topic_count, written_by_count, count(r) AS has_topic_count
        RETURN paper_count, author_count, topic_count, written_by_count, has_topic_count
        """
    )
    if not result:
        return GraphStats(
            paper_count=0,
            author_count=0,
            topic_count=0,
            written_by_count=0,
            has_topic_count=0,
        )
    row = result[0]
    return GraphStats(**row)


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    client: Neo4jClient = app.state.neo4j
    retrieval = retrieve_papers(client, request.query)
    if not retrieval.papers:
        return QueryResponse(
            answer="Insufficient information in the current knowledge graph.",
            papers=[],
        )
    answer = synthesize_answer(request.query, retrieval.papers)
    papers = [Paper(**p) for p in retrieval.papers]
    return QueryResponse(answer=answer, papers=papers)
