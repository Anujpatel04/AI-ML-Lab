from __future__ import annotations

from typing import Any
import requests
from .config import settings


INSUFFICIENT_MESSAGE = "Insufficient information in the current knowledge graph."

SYSTEM_PROMPT = (
    "You are a research assistant. Use ONLY the provided paper abstracts and metadata. "
    "Do not invent papers, authors, or facts. If the context is empty, respond exactly: "
    f"{INSUFFICIENT_MESSAGE}"
)


def _build_context(papers: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for p in papers:
        authors = ", ".join(p.get("authors", []))
        topics = ", ".join(p.get("topics", []))
        parts.append(
            f"Title: {p.get('title')}\n"
            f"Year: {p.get('year')}\n"
            f"Authors: {authors}\n"
            f"Topics: {topics}\n"
            f"Abstract: {p.get('abstract')}"
        )
    return "\n\n".join(parts)


def _fallback_summary(papers: list[dict[str, Any]], max_items: int = 5) -> str:
    lines = [
        "Summary based on retrieved papers:",
    ]
    for p in papers[:max_items]:
        authors = ", ".join(p.get("authors", []))
        lines.append(f"- {p.get('title')} ({p.get('year')}) — {authors}")
    return "\n".join(lines)


def synthesize_answer(query: str, papers: list[dict[str, Any]]) -> str:
    if not papers:
        return INSUFFICIENT_MESSAGE

    if not settings.openai_api_key:
        context = _build_context(papers)
        return "LLM not configured. Retrieved context:\n\n" + context[:2000]

    context = _build_context(papers)
    payload = {
        "model": settings.openai_model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"},
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {settings.openai_api_key}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        f"{settings.openai_base_url}/chat/completions",
        headers=headers,
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()
    content = data["choices"][0]["message"]["content"].strip()
    if content == INSUFFICIENT_MESSAGE:
        return _fallback_summary(papers)
    return content
