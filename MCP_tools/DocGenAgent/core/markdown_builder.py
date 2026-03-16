"""
Build structured markdown sections: Overview, Installation, Usage, API Reference.
"""

from typing import List

from core.code_parser import FileMeta, FunctionMeta


def section_project_overview(overview_text: str) -> str:
    """Build the Project Overview section."""
    return f"""## Project Overview

{overview_text}
"""


def section_installation() -> str:
    """Build a generic Installation section placeholder."""
    return """## Installation

```bash
pip install -r requirements.txt
```
"""


def section_usage() -> str:
    """Build a generic Usage section placeholder."""
    return """## Usage

Run the application or import the package as needed. See API Reference for details.
"""


def section_api_reference(api_content: str) -> str:
    """Wrap API documentation in an API Reference section."""
    return f"""## API Reference

{api_content}
"""


def section_table_of_contents(toc_markdown: str) -> str:
    """Build table of contents section."""
    return f"""## Table of Contents

{toc_markdown}
"""


def build_structure_summary(file_metas: List[FileMeta]) -> str:
    """Build a short text summary of code structure for the LLM overview."""
    lines: List[str] = []
    for fm in file_metas:
        lines.append(f"File: {fm.file_name}")
        for c in fm.classes:
            lines.append(f"  Class: {c.name}")
            for m in c.methods:
                lines.append(f"    - {m.name}({', '.join(m.parameters)})")
        for f in fm.functions:
            lines.append(f"  Function: {f.name}({', '.join(f.parameters)})")
    return "\n".join(lines) if lines else "No public API found."


def format_anchor(title: str) -> str:
    """Convert a section title to a markdown anchor (lowercase, spaces to hyphens)."""
    return title.lower().replace(" ", "-").replace("(", "").replace(")", "")
