"""
Connect to Neo4j and query the knowledge graph for entities and relationships.
Returns readable context for the LLM. Tolerates missing/empty graph.
Uses connection_timeout so the app does not hang if Neo4j is down.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import List, Optional

logger = logging.getLogger(__name__)

# Seconds to wait for Neo4j connection and query (avoid infinite hang)
NEO4J_CONNECTION_TIMEOUT = 5
NEO4J_QUERY_TIMEOUT = 10


def get_driver(uri: str, username: str, password: str, connection_timeout: float = NEO4J_CONNECTION_TIMEOUT):
    """Create and return a Neo4j driver. Caller must close it."""
    from neo4j import GraphDatabase
    return GraphDatabase.driver(uri, auth=(username, password), connection_timeout=connection_timeout)


def _query_graph_impl(uri: str, username: str, password: str, entity_names: List[str], limit: int) -> str:
    """Inner graph query (runs in thread for timeout)."""
    driver = None
    try:
        driver = get_driver(uri, username, password)
        with driver.session() as session:
            # Flexible: match nodes that have name/title in our list, and their relationships
            names = [n.strip() for n in entity_names if n.strip()][:20]
            if not names:
                return ""
            # Cypher: match (a)-[r]-(b) where a or b has name/title in list.
            # Use keys() so we don't reference missing properties (avoids Neo4j schema warnings when DB is empty).
            result = session.run(
                """
                MATCH (a)-[r]->(b)
                WHERE (
                    ('name' IN keys(a) AND toLower(trim(toString(a.name))) IN $names_lower)
                    OR ('name' IN keys(b) AND toLower(trim(toString(b.name))) IN $names_lower)
                    OR ('title' IN keys(a) AND toLower(trim(toString(a.title))) IN $names_lower)
                    OR ('title' IN keys(b) AND toLower(trim(toString(b.title))) IN $names_lower)
                )
                RETURN a, type(r) AS rel_type, b
                LIMIT $limit
                """,
                names_lower=[n.lower() for n in names],
                limit=limit,
            )
            lines = []
            seen = set()
            for record in result:
                a, rel_type, b = record.get("a"), record.get("rel_type"), record.get("b")
                if a is None or b is None:
                    continue
                a_name = _node_label(a)
                b_name = _node_label(b)
                key = (a_name, rel_type, b_name)
                if key in seen:
                    continue
                seen.add(key)
                lines.append(f"  ({a_name}) -[{rel_type}]-> ({b_name})")
            if lines:
                return "Knowledge graph (entities and relationships):\n" + "\n".join(lines)
            return ""
    except Exception as e:
        logger.warning("Graph query failed: %s", e)
        return ""
    finally:
        if driver:
            try:
                driver.close()
            except Exception:
                pass


def query_graph(
    uri: str,
    username: str,
    password: str,
    entity_names: List[str],
    *,
    limit: int = 20,
    query_timeout: float = NEO4J_QUERY_TIMEOUT,
) -> str:
    """
    Query Neo4j for nodes and relationships involving the given entity names.
    Uses a timeout so the app does not hang if Neo4j is slow or down.
    Returns a single string of graph context for the LLM, or "" on timeout/error.
    """
    if not entity_names:
        return ""
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_query_graph_impl, uri, username, password, entity_names, limit)
            return fut.result(timeout=query_timeout)
    except FuturesTimeoutError:
        logger.warning("Neo4j query timed out after %s seconds (is Neo4j running?)", query_timeout)
        return ""
    except Exception as e:
        logger.warning("Graph query failed: %s", e)
        return ""


def _node_label(node) -> str:
    """Get a short label for a Neo4j node for display. Node may be neo4j.Node or dict-like."""
    if node is None:
        return "?"
    getter = getattr(node, "get", None)
    if callable(getter):
        for key in ("name", "title", "id"):
            try:
                v = getter(key)
            except Exception:
                v = None
            if v is not None and str(v).strip():
                return str(v).strip()[:80]
    if hasattr(node, "labels"):
        try:
            labels = list(node.labels)
            if labels:
                return labels[0]
        except Exception:
            pass
    return "Node"
