"""LangChain: analyze issue and parse structured JSON."""

import json
import re
from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from llm.prompts import ISSUE_ANALYSIS_HUMAN, ISSUE_ANALYSIS_SYSTEM
from models.schemas import IssueAnalysisResult, IssueInput
from utils.config import (
    AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_BASE_URL,
    AZURE_OPENAI_DEPLOYMENT,
    AZURE_OPENAI_API_VERSION,
    LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_API_KEY,
)
from utils.logger import get_logger

logger = get_logger(__name__)


def _get_llm():
    if LLM_PROVIDER == "ollama":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
    from langchain_openai import AzureChatOpenAI
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_BASE_URL,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    )


def _format_optional_list(items: list[str] | None, empty: str = "none") -> str:
    if not items:
        return empty
    return "\n".join(f"- {c}" for c in items)


def _format_metadata(meta: dict[str, str] | None) -> str:
    if not meta:
        return "none"
    return "\n".join(f"{k}: {v}" for k, v in meta.items())


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse JSON from model output; strip markdown fences if present."""
    raw = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if fence:
        raw = fence.group(1).strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    return json.loads(raw[start : end + 1])


def analyze_issue(issue: IssueInput) -> IssueAnalysisResult:
    """Run LLM and validate against IssueAnalysisResult."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ISSUE_ANALYSIS_SYSTEM),
            ("human", ISSUE_ANALYSIS_HUMAN),
        ]
    )
    chain = prompt | _get_llm() | StrOutputParser()
    text = chain.invoke(
        {
            "title": issue.title,
            "description": issue.description or "none",
            "comments": _format_optional_list(issue.comments),
            "labels": _format_optional_list(issue.labels),
            "metadata": _format_metadata(issue.metadata),
        }
    )
    logger.info("issue_analyzer | llm_response_length=%s", len(text))
    data = _extract_json_object(text)
    return IssueAnalysisResult.model_validate(data)
