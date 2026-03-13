"""
Scoring utilities for interview evaluation.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def normalize_score(value: Any, min_score: int = 1, max_score: int = 10) -> int:
    """Coerce a value to an integer score within [min_score, max_score]."""
    if isinstance(value, int):
        return max(min_score, min(max_score, value))
    if isinstance(value, (float, str)):
        try:
            n = int(float(value))
            return max(min_score, min(max_score, n))
        except (ValueError, TypeError):
            pass
    return (min_score + max_score) // 2


def extract_json_from_text(text: str) -> dict | None:
    """Extract first JSON object from LLM output (handles nested braces)."""
    if not text or not isinstance(text, str):
        return None
    text = text.strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    import json
                    return json.loads(text[start : i + 1])
                except json.JSONDecodeError:
                    break
    return None
