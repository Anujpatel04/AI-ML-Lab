"""
Orchestrates documentation generation: combines repo read, parse, LLM, and markdown.
"""

from typing import List

from core.code_parser import FileMeta, parse_repository
from core.repo_reader import CodeFile, read_repository
from core.markdown_builder import (
    build_structure_summary,
    section_api_reference,
    section_installation,
    section_project_overview,
    section_table_of_contents,
    section_usage,
)
from core.file_writer import write_readme
from llm.chains import generate_class_doc, generate_function_doc, generate_overview, generate_table_of_contents
from utils.logger import get_logger, log_docs_generated

logger = get_logger(__name__)


def _generate_api_docs_for_file(file_meta: FileMeta) -> str:
    """Generate API reference markdown for one file (classes + functions)."""
    parts: List[str] = []
    for c in file_meta.classes:
        methods_block = "\n".join(
            f"- {m.name}({', '.join(m.parameters)}) -> {m.returns}; doc: {m.docstring[:80]}..."
            for m in c.methods
        )
        doc = generate_class_doc(c.name, c.docstring, methods_block)
        parts.append(doc)
    for f in file_meta.functions:
        doc = generate_function_doc(
            name=f.name,
            parameters=f.parameters,
            returns=f.returns,
            docstring=f.docstring,
        )
        parts.append(doc)
    return "\n\n".join(parts)


def generate_documentation(repo_path: str) -> str:
    """
    Full pipeline: read repo, parse, generate docs with LLM, build markdown, write README.
    Returns the path to the generated README.
    """
    code_files: List[CodeFile] = read_repository(repo_path)
    if not code_files:
        logger.warning("doc_generator | no_code_files | repo_path=%s", repo_path)

    file_metas: List[FileMeta] = parse_repository(code_files)
    structure_summary = build_structure_summary(file_metas)

    overview_text = generate_overview(structure_summary)
    sections = [
        section_project_overview(overview_text),
        section_installation(),
        section_usage(),
    ]

    headings = ["Project Overview", "Installation", "Usage", "API Reference"]
    toc = generate_table_of_contents(headings)
    sections.insert(1, section_table_of_contents(toc))

    api_parts: List[str] = []
    for fm in file_metas:
        file_doc = _generate_api_docs_for_file(fm)
        if file_doc:
            api_parts.append(f"### {fm.file_name}\n\n{file_doc}")
    api_section = section_api_reference("\n\n".join(api_parts) if api_parts else "No API items extracted.")
    sections.append(api_section)

    full_markdown = "# Project Documentation\n\n" + "\n".join(sections)
    readme_path = write_readme(full_markdown)
    log_docs_generated(logger, str(readme_path))
    return str(readme_path)
