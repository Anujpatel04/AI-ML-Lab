from __future__ import annotations

import logging
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Iterable

from ..config import settings
from ..graph import Neo4jClient
from .category_map import CATEGORY_MAP


ATOM_NS = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

logger = logging.getLogger(__name__)


def fetch_arxiv_xml(query: str, max_results: int) -> str:
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
    }
    url = "http://export.arxiv.org/api/query?" + urllib.parse.urlencode(params)
    logger.info("Fetching arXiv data: %s", url)
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def _map_topics(categories: Iterable[str]) -> list[str]:
    topics = []
    for c in categories:
        if c in CATEGORY_MAP:
            topics.append(CATEGORY_MAP[c])
        else:
            topics.append(c.replace(".", " ").replace("-", " ").title())
    return sorted(set(topics))


def parse_arxiv(xml_data: str) -> list[dict[str, object]]:
    root = ET.fromstring(xml_data)
    entries = []
    for entry in root.findall(f"{ATOM_NS}entry"):
        arxiv_id = entry.findtext(f"{ATOM_NS}id", default="").split("/")[-1]
        title = (entry.findtext(f"{ATOM_NS}title", default="") or "").strip()
        summary = (entry.findtext(f"{ATOM_NS}summary", default="") or "").strip()
        published = entry.findtext(f"{ATOM_NS}published", default="")
        year = int(published[:4]) if published else 0
        authors = [
            (a.findtext(f"{ATOM_NS}name", default="") or "").strip()
            for a in entry.findall(f"{ATOM_NS}author")
        ]
        categories = [
            c.attrib.get("term", "")
            for c in entry.findall(f"{ATOM_NS}category")
        ]
        topics = _map_topics(categories)
        entries.append(
            {
                "id": arxiv_id,
                "title": title,
                "abstract": summary,
                "year": year,
                "authors": [a for a in authors if a],
                "topics": topics,
            }
        )
    return entries


def ingest() -> None:
    logger.info("Starting ingestion")
    xml_data = fetch_arxiv_xml(settings.arxiv_query, settings.arxiv_max_results)
    records = parse_arxiv(xml_data)
    if not records:
        logger.warning("No records found from arXiv")
        return

    client = Neo4jClient()
    try:
        client.create_constraints()
        for r in records:
            client.merge_paper_with_links(
                {"id": r["id"], "title": r["title"], "year": r["year"], "abstract": r["abstract"]},
                r["authors"],
                r["topics"],
            )
        logger.info("Ingested %d papers", len(records))
    finally:
        client.close()


if __name__ == "__main__":
    logging.basicConfig(level=settings.log_level)
    ingest()
