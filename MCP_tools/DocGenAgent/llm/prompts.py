"""
Prompt templates for the LLM documentation generator.
"""

DOC_SECTION_SYSTEM = """You are a technical writer. Generate clear, professional documentation for code.
Output only the requested section in valid markdown. No preamble or meta-commentary."""

DOC_FUNCTION_TEMPLATE = """Generate documentation for the following function. Output markdown only.

Function name: {name}
Parameters: {parameters}
Returns: {returns}
Docstring (if any): {docstring}

Include:
1. A short description (one or two sentences).
2. **Parameters:** bullet list with name and type/purpose.
3. **Returns:** what is returned.
4. **Example:** a minimal code example in a fenced code block.

Format the heading as: ### {name}()
"""

DOC_CLASS_TEMPLATE = """Generate documentation for the following class and its methods. Output markdown only.

Class name: {name}
Class docstring: {docstring}
Methods:
{methods_block}

Include:
1. A short class description.
2. For each method: ### method_name() with description, parameters, returns, and a brief example if useful.
"""

OVERVIEW_TEMPLATE = """Based on the following code structure summary, write a concise **Project Overview** section (2–4 sentences) for a README. Output only the overview paragraph, no heading.

Code structure:
{structure_summary}
"""

TABLE_OF_CONTENTS_TEMPLATE = """Given these documentation section headings (as used in the README), output a markdown **Table of Contents** with anchor links. Use - [Section Name](#section-name) format. Output only the list, no extra text.

Sections:
{headings}
"""
