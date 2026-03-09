"""
Graph retriever using Neo4j for entity and relationship queries.
"""

from typing import Any

from openai import AzureOpenAI

from config import (
    AZURE_BASE_URL,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    GRAPH_LIMIT,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
)
from utils.logger import get_logger

logger = get_logger(__name__)

ENTITY_SYSTEM = """Extract entity names from the user question for a knowledge graph search.
Return a JSON object: {"entities": ["name1", "name2", ...]}
Use short, canonical names. If none found, return {"entities": []}. Output only JSON."""


def _get_llm_client() -> AzureOpenAI:
    if not AZURE_OPENAI_API_KEY or not AZURE_BASE_URL:
        raise ValueError("Azure credentials not set")
    return AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_BASE_URL,
        api_version=AZURE_OPENAI_API_VERSION,
    )


def _extract_entities(query: str) -> list[str]:
    """Use LLM to extract entity names from query."""
    import json
    import re
    query = (query or "").strip()
    if not query:
        return []
    try:
        client = _get_llm_client()
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": ENTITY_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=128,
        )
        text = (resp.choices[0].message.content or "").strip()
        match = re.search(r"\{[^{}]*\}", text)
        if match:
            data = json.loads(match.group())
            entities = data.get("entities", [])
            if isinstance(entities, list):
                return [str(e).strip() for e in entities if str(e).strip()][:10]
    except Exception as e:
        logger.warning("Entity extraction failed: %s", e)
    return []


def retrieve(query: str, limit: int | None = None) -> str:
    """
    Retrieve graph context from Neo4j. Returns readable text context.
    """
    limit = limit or GRAPH_LIMIT
    query = (query or "").strip()
    if not query:
        return ""
    try:
        from neo4j import GraphDatabase
    except ImportError:
        logger.warning("neo4j not installed; install with: pip install neo4j")
        return ""
    entities = _extract_entities(query)
    if not entities:
        return ""
    driver = None
    try:
        driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USERNAME, NEO4J_PASSWORD),
            connection_timeout=5,
        )
        with driver.session() as session:
            # Match nodes and relationships involving entity-like names
            names_lower = [e.lower() for e in entities]
            result = session.run(
                """
                MATCH (n)-[r]->(m)
                WHERE ('name' IN keys(n) AND toLower(toString(n.name)) IN $names_lower)
                   OR ('name' IN keys(m) AND toLower(toString(m.name)) IN $names_lower)
                   OR ('title' IN keys(n) AND toLower(toString(n.title)) IN $names_lower)
                   OR ('title' IN keys(m) AND toLower(toString(m.title)) IN $names_lower)
                RETURN n, type(r) AS rel_type, m
                LIMIT $limit
                """,
                names_lower=names_lower,
                limit=limit,
            )
            lines = []
            seen = set()
            for record in result:
                n, rtype, m = record.get("n"), record.get("rel_type"), record.get("m")
                if n is None or m is None:
                    continue
                na = _node_label(n)
                ma = _node_label(m)
                key = (na, rtype, ma)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"  ({na}) -[{rtype}]-> ({ma})")
            if lines:
                return "Knowledge graph:\n" + "\n".join(lines)
            return ""
    except Exception as e:
        logger.warning("Graph retrieval failed: %s", e)
        return ""
    finally:
        if driver:
            try:
                driver.close()
            except Exception:
                pass


def _node_label(node: Any) -> str:
    if node is None:
        return "?"
    for key in ("name", "title", "id"):
        try:
            v = node.get(key) if hasattr(node, "get") else getattr(node, key, None)
            if v is not None and str(v).strip():
                return str(v).strip()[:80]
        except Exception:
            pass
    if hasattr(node, "labels"):
        try:
            labels = list(node.labels)
            if labels:
                return labels[0]
        except Exception:
            pass
    return "Node"
