from __future__ import annotations

from typing import Any, Iterable
from neo4j import GraphDatabase, Driver
from .config import settings


class Neo4jClient:
    def __init__(self) -> None:
        self._driver: Driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )

    def close(self) -> None:
        self._driver.close()

    def run(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self._driver.session() as session:
            result = session.run(query, params or {})
            return [record.data() for record in result]

    def create_constraints(self) -> None:
        constraint_queries = [
            "CREATE CONSTRAINT paper_id_unique IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT author_name_unique IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
            "CREATE CONSTRAINT topic_name_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.name IS UNIQUE",
        ]
        for q in constraint_queries:
            self.run(q)

    def merge_paper_with_links(
        self,
        paper: dict[str, Any],
        authors: Iterable[str],
        topics: Iterable[str],
    ) -> None:
        query = """
        MERGE (p:Paper {id: $id})
        SET p.title = $title,
            p.year = $year,
            p.abstract = $abstract
        WITH p
        UNWIND $authors AS author_name
        MERGE (a:Author {name: author_name})
        MERGE (p)-[:WRITTEN_BY]->(a)
        WITH p
        UNWIND $topics AS topic_name
        MERGE (t:Topic {name: topic_name})
        MERGE (p)-[:HAS_TOPIC]->(t)
        """
        params = {
            "id": paper["id"],
            "title": paper["title"],
            "year": paper["year"],
            "abstract": paper["abstract"],
            "authors": list(authors),
            "topics": list(topics),
        }
        self.run(query, params)
