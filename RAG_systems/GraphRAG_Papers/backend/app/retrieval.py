from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from .graph import Neo4jClient


@dataclass
class RetrievalResult:
    papers: list[dict[str, Any]]
    matched_entities: dict[str, list[str]]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def extract_entities(query: str) -> dict[str, list[str]]:
    q = query.lower()
    entities: dict[str, list[str]] = {"authors": [], "topics": [], "papers": []}

    author_match = re.search(r"by\s+([a-z][a-z\s\-\.]+)", q)
    if author_match:
        entities["authors"].append(_normalize(author_match.group(1)))

    title_match = re.search(r"\"(.+?)\"", query)
    if title_match:
        entities["papers"].append(_normalize(title_match.group(1)))

    topic_keywords = [
        "graph neural networks",
        "gnn",
        "knowledge graph",
        "graph learning",
        "graph representation",
        "graph embeddings",
    ]
    for kw in topic_keywords:
        if kw in q:
            entities["topics"].append(kw)

    if "author" in q and not entities["authors"]:
        entities["authors"].append(_normalize(query))

    return entities


def retrieve_papers(client: Neo4jClient, query: str) -> RetrievalResult:
    entities = extract_entities(query)
    papers: list[dict[str, Any]] = []

    if entities["authors"]:
        author = entities["authors"][0]
        cypher = """
        MATCH (a:Author)
        WHERE toLower(a.name) CONTAINS toLower($author)
        MATCH (p:Paper)-[:WRITTEN_BY]->(a)
        OPTIONAL MATCH (p)-[:WRITTEN_BY]->(a2:Author)
        OPTIONAL MATCH (p)-[:HAS_TOPIC]->(t:Topic)
        RETURN p.id AS id,
               p.title AS title,
               p.year AS year,
               p.abstract AS abstract,
               collect(DISTINCT a2.name) AS authors,
               collect(DISTINCT t.name) AS topics
        ORDER BY p.year DESC
        LIMIT 20
        """
        papers = client.run(cypher, {"author": author})

    if not papers and entities["topics"]:
        topic = entities["topics"][0]
        cypher = """
        MATCH (t:Topic)
        WHERE toLower(t.name) CONTAINS toLower($topic)
        MATCH (p:Paper)-[:HAS_TOPIC]->(t)
        OPTIONAL MATCH (p)-[:WRITTEN_BY]->(a:Author)
        OPTIONAL MATCH (p)-[:HAS_TOPIC]->(t2:Topic)
        RETURN p.id AS id,
               p.title AS title,
               p.year AS year,
               p.abstract AS abstract,
               collect(DISTINCT a.name) AS authors,
               collect(DISTINCT t2.name) AS topics
        ORDER BY p.year DESC
        LIMIT 20
        """
        papers = client.run(cypher, {"topic": topic})

    if not papers and entities["papers"]:
        title = entities["papers"][0]
        cypher = """
        MATCH (p:Paper)
        WHERE toLower(p.title) CONTAINS toLower($title)
        OPTIONAL MATCH (p)-[:WRITTEN_BY]->(a:Author)
        OPTIONAL MATCH (p)-[:HAS_TOPIC]->(t:Topic)
        RETURN p.id AS id,
               p.title AS title,
               p.year AS year,
               p.abstract AS abstract,
               collect(DISTINCT a.name) AS authors,
               collect(DISTINCT t.name) AS topics
        ORDER BY p.year DESC
        LIMIT 10
        """
        papers = client.run(cypher, {"title": title})

    if not papers:
        cypher = """
        MATCH (p:Paper)
        OPTIONAL MATCH (p)-[:WRITTEN_BY]->(a:Author)
        OPTIONAL MATCH (p)-[:HAS_TOPIC]->(t:Topic)
        RETURN p.id AS id,
               p.title AS title,
               p.year AS year,
               p.abstract AS abstract,
               collect(DISTINCT a.name) AS authors,
               collect(DISTINCT t.name) AS topics
        ORDER BY p.year DESC
        LIMIT 5
        """
        papers = client.run(cypher)

    return RetrievalResult(papers=papers, matched_entities=entities)
