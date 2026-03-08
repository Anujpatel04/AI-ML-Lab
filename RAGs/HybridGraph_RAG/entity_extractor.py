"""
Extract entities (people, organizations, products, topics) from user queries.
Uses an LLM for quality; falls back to empty list on failure.
"""

import json
import logging
from typing import List

logger = logging.getLogger(__name__)

ENTITY_SYSTEM = """You extract key entities from the user's question for knowledge-graph and document search.
Output a JSON object with one key "entities" whose value is a list of strings.
Include: person names, organizations, products, and main topics/concepts.
Use short, canonical forms (e.g. "Microsoft" not "Microsoft Corp").
If there are no clear entities, return {"entities": []}.
Output only valid JSON, no other text."""


def extract_entities(client, model: str, query: str) -> List[str]:
    """
    Extract entity strings from a user query using the LLM.

    Args:
        client: OpenAI or AzureOpenAI client.
        model: Chat model or deployment name.
        query: User question.

    Returns:
        List of entity strings (e.g. person names, orgs, products, topics). Empty on error.
    """
    if not (query or "").strip():
        return []
    query = query.strip()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ENTITY_SYSTEM},
                {"role": "user", "content": query},
            ],
            temperature=0,
            max_tokens=256,
        )
        text = (resp.choices[0].message.content or "").strip()
        data = _parse_entity_response(text)
        entities = list(data) if isinstance(data, list) else data.get("entities", [])
        if isinstance(entities, list):
            return [str(e).strip() for e in entities if str(e).strip()]
        return []
    except Exception as e:
        logger.warning("Entity extraction failed: %s", e)
        return []


def _parse_entity_response(text: str) -> dict:
    """Parse LLM JSON; tolerate markdown code blocks."""
    text = text.strip()
    if "```" in text:
        start = text.find("```")
        if start >= 0:
            start = text.find("\n", start) + 1
            end = text.find("```", start)
            if end > start:
                text = text[start:end]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"entities": []}
