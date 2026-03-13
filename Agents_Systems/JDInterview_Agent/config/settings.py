"""
Configuration for Adaptive AI Interview Simulator. Loads from repo root .env.
"""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DOTENV_PATH = REPO_ROOT / ".env"
if DOTENV_PATH.is_file():
    from dotenv import load_dotenv
    load_dotenv(DOTENV_PATH)


def _parse_azure_endpoint(url: str) -> tuple[str, str]:
    url = (url or "").strip()
    if "/openai/deployments/" in url:
        base = url.split("/openai/deployments/")[0].strip("/")
        rest = url.split("/openai/deployments/")[1]
        deployment = (rest.split("/")[0].split("?")[0].strip() if rest else "") or "gpt-4o"
        return base, deployment
    return url.strip("/"), "gpt-4o"


AZURE_OPENAI_API_KEY = (
    os.environ.get("AZURE_OPENAI_API_KEY") or os.environ.get("AZURE_KEY") or ""
).strip()
AZURE_OPENAI_ENDPOINT_RAW = (
    os.environ.get("AZURE_OPENAI_ENDPOINT") or os.environ.get("AZURE_ENDPOINT") or ""
).strip()
AZURE_OPENAI_API_VERSION = (
    os.environ.get("AZURE_OPENAI_API_VERSION") or os.environ.get("API_VERSION") or "2024-02-15-preview"
).strip()
AZURE_OPENAI_BASE_URL, AZURE_OPENAI_DEPLOYMENT = _parse_azure_endpoint(AZURE_OPENAI_ENDPOINT_RAW)

DEFAULT_QUESTION_LIMIT = int(os.environ.get("INTERVIEW_QUESTION_LIMIT", "5"))
LOG_LEVEL = (os.environ.get("LOG_LEVEL") or "INFO").strip()
