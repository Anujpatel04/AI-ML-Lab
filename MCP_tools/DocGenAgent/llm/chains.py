"""
LangChain chains for generating documentation with Azure OpenAI or Ollama.
"""

from typing import List

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

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
from utils.logger import get_logger, log_llm_call

from llm.prompts import (
    DOC_CLASS_TEMPLATE,
    DOC_FUNCTION_TEMPLATE,
    OVERVIEW_TEMPLATE,
    TABLE_OF_CONTENTS_TEMPLATE,
)

logger = get_logger(__name__)


def _get_llm():
    """Build LLM instance: Azure OpenAI, OpenAI, or Ollama."""
    if LLM_PROVIDER == "ollama":
        from langchain_community.chat_models import ChatOllama
        log_llm_call(logger, "ollama", {"model": OLLAMA_MODEL})
        return ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    if LLM_PROVIDER == "openai" and OPENAI_API_KEY:
        from langchain_openai import ChatOpenAI
        log_llm_call(logger, "openai", {})
        return ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o-mini")
    # Default: Azure OpenAI
    from langchain_openai import AzureChatOpenAI
    log_llm_call(logger, "azure", {"deployment": AZURE_OPENAI_DEPLOYMENT})
    return AzureChatOpenAI(
        azure_endpoint=AZURE_OPENAI_BASE_URL,
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    )


def generate_function_doc(name: str, parameters: List[str], returns: str, docstring: str) -> str:
    """Generate markdown documentation for a single function."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a technical writer. Output only the requested markdown, no preamble."),
        ("human", DOC_FUNCTION_TEMPLATE),
    ])
    chain = prompt | _get_llm() | StrOutputParser()
    result = chain.invoke({
        "name": name,
        "parameters": ", ".join(parameters) if parameters else "none",
        "returns": returns or "None",
        "docstring": docstring or "No docstring.",
    })
    return (result or "").strip()


def generate_class_doc(name: str, docstring: str, methods_block: str) -> str:
    """Generate markdown documentation for a class and its methods."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a technical writer. Output only the requested markdown, no preamble."),
        ("human", DOC_CLASS_TEMPLATE),
    ])
    chain = prompt | _get_llm() | StrOutputParser()
    result = chain.invoke({
        "name": name,
        "docstring": docstring or "No docstring.",
        "methods_block": methods_block,
    })
    return (result or "").strip()


def generate_overview(structure_summary: str) -> str:
    """Generate project overview paragraph."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a technical writer. Output only the overview paragraph."),
        ("human", OVERVIEW_TEMPLATE),
    ])
    chain = prompt | _get_llm() | StrOutputParser()
    result = chain.invoke({"structure_summary": structure_summary})
    return (result or "").strip()


def generate_table_of_contents(headings: List[str]) -> str:
    """Generate markdown table of contents with anchor links."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Output only the markdown list for table of contents."),
        ("human", TABLE_OF_CONTENTS_TEMPLATE),
    ])
    section_list = "\n".join(headings)
    chain = prompt | _get_llm() | StrOutputParser()
    result = chain.invoke({"headings": section_list})
    return (result or "").strip()
