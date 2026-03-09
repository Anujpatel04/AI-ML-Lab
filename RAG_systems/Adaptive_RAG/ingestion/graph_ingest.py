"""
Graph ingestion: parse documents, extract entities, insert nodes and relationships into Neo4j.
"""

import json
import re
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from openai import AzureOpenAI

from config import (
    AZURE_BASE_URL,
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_API_VERSION,
    AZURE_OPENAI_DEPLOYMENT,
    DATA_DIR,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
)
from utils.logger import get_logger

logger = get_logger(__name__)

EXTRACT_SYSTEM = """From the following text, extract entities and relationships.
For each relationship, output: (source_entity, relationship_type, target_entity).
Use short entity names. Output a JSON array of objects: {"s": "source", "r": "relationship", "t": "target"}.
If none found, output []. Output only the JSON array."""


def load_documents(data_dir: Path | None = None) -> list[str]:
    """Load text from .txt and .md files."""
    data_dir = Path(data_dir or DATA_DIR)
    if not data_dir.is_dir():
        return []
    texts = []
    for path in sorted(data_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in (".txt", ".md"):
            try:
                texts.append(path.read_text(encoding="utf-8", errors="replace"))
            except Exception as e:
                logger.warning("Failed to read %s: %s", path, e)
    return texts


def _extract_triples(text: str, client: AzureOpenAI) -> list[tuple[str, str, str]]:
    """Use LLM to extract (source, relation, target) triples."""
    if not text.strip():
        return []
    try:
        resp = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT,
            messages=[
                {"role": "system", "content": EXTRACT_SYSTEM},
                {"role": "user", "content": text[:6000]},
            ],
            temperature=0.0,
            max_tokens=1024,
        )
        raw = (resp.choices[0].message.content or "").strip()
        match = re.search(r"\[[\s\S]*\]", raw)
        if not match:
            return []
        arr = json.loads(match.group())
        out = []
        for item in arr:
            if isinstance(item, dict):
                s = (item.get("s") or item.get("source") or "").strip()
                r = (item.get("r") or item.get("relationship") or "RELATES_TO").strip() or "RELATES_TO"
                t = (item.get("t") or item.get("target") or "").strip()
                if s and t:
                    out.append((s, r, t))
        return out
    except Exception as e:
        logger.warning("Triple extraction failed: %s", e)
        return []


def run(data_dir: Path | None = None) -> None:
    """Load docs, extract triples, insert into Neo4j."""
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise ImportError("neo4j required: pip install neo4j") from e
    if not AZURE_OPENAI_API_KEY or not AZURE_BASE_URL:
        raise ValueError("Azure credentials not set")
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_BASE_URL,
        api_version=AZURE_OPENAI_API_VERSION,
    )
    docs = load_documents(data_dir)
    if not docs:
        logger.warning("No documents loaded")
        return
    all_triples = []
    for doc in docs:
        all_triples.extend(_extract_triples(doc, client))
    if not all_triples:
        logger.warning("No triples extracted")
        return
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD), connection_timeout=10)
    try:
        with driver.session() as session:
            for s, r, t in all_triples:
                rel = re.sub(r"[^a-zA-Z0-9_]", "_", r)[:50] or "RELATES_TO"
                if not rel[0].isalpha():
                    rel = "REL_" + rel
                session.run(
                    """
                    MERGE (a:Entity {name: $s})
                    MERGE (b:Entity {name: $t})
                    MERGE (a)-[r:%s]->(b)
                    """ % rel,
                    s=s[:500],
                    t=t[:500],
                )
        logger.info("Inserted %d triples into Neo4j", len(all_triples))
    finally:
        driver.close()


if __name__ == "__main__":
    run()
    sys.exit(0)
