"""LLM chains and prompts for documentation generation."""

from llm.chains import (
    generate_class_doc,
    generate_function_doc,
    generate_overview,
    generate_table_of_contents,
)

__all__ = [
    "generate_class_doc",
    "generate_function_doc",
    "generate_overview",
    "generate_table_of_contents",
]
