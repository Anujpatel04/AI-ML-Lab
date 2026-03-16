"""
Code parser: uses Python AST to extract functions, classes, signatures, and docstrings.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

from utils.logger import get_logger, log_files_processed

from core.repo_reader import CodeFile

logger = get_logger(__name__)


@dataclass
class FunctionMeta:
    """Metadata for a single function or method."""

    name: str
    parameters: List[str]
    returns: str
    docstring: str
    is_method: bool = False


@dataclass
class ClassMeta:
    """Metadata for a class and its methods."""

    name: str
    docstring: str
    methods: List[FunctionMeta] = field(default_factory=list)


@dataclass
class FileMeta:
    """Structured metadata for one file: classes and top-level functions."""

    file_name: str
    classes: List[ClassMeta] = field(default_factory=list)
    functions: List[FunctionMeta] = field(default_factory=list)


def _get_docstring(node: ast.AST) -> str:
    """Extract docstring from an AST node."""
    doc = ast.get_docstring(node)
    return (doc or "").strip()


def _get_returns_annotation(node: ast.FunctionDef) -> str:
    """Get return annotation as string."""
    if node.returns is None:
        return ""
    if isinstance(node.returns, ast.Constant):
        return str(node.returns.value)
    return ast.unparse(node.returns) if hasattr(ast, "unparse") else ""


def _get_parameter_names(node: ast.FunctionDef) -> List[str]:
    """Get list of parameter names (excluding *args, **kwargs placeholders)."""
    names: List[str] = []
    for arg in node.args.args:
        if arg.arg in ("self", "cls"):
            continue
        names.append(arg.arg)
    return names


def _parse_functions_and_classes(tree: ast.AST) -> tuple[List[ClassMeta], List[FunctionMeta]]:
    """Walk module AST and collect classes and top-level functions."""
    classes: List[ClassMeta] = []
    functions: List[FunctionMeta] = []

    if not isinstance(tree, ast.Module):
        return classes, functions
    for child in tree.body:
        if isinstance(child, ast.ClassDef):
            methods: List[FunctionMeta] = []
            for item in child.body:
                if isinstance(item, ast.FunctionDef):
                    methods.append(
                        FunctionMeta(
                            name=item.name,
                            parameters=_get_parameter_names(item),
                            returns=_get_returns_annotation(item),
                            docstring=_get_docstring(item),
                            is_method=True,
                        )
                    )
            classes.append(
                ClassMeta(
                    name=child.name,
                    docstring=_get_docstring(child),
                    methods=methods,
                )
            )
        elif isinstance(child, ast.FunctionDef):
            functions.append(
                FunctionMeta(
                    name=child.name,
                    parameters=_get_parameter_names(child),
                    returns=_get_returns_annotation(child),
                    docstring=_get_docstring(child),
                    is_method=False,
                )
            )
    return classes, functions


def parse_code_file(code_file: CodeFile) -> FileMeta | None:
    """
    Parse a single code file with AST and return structured metadata.

    Returns None if the file cannot be parsed (e.g. syntax error).
    """
    try:
        tree = ast.parse(code_file.content)
    except SyntaxError as e:
        logger.warning("code_parser | parse_error | file=%s | error=%s", code_file.path, e)
        return None

    classes, functions = _parse_functions_and_classes(tree)
    file_name = Path(code_file.path).name
    log_files_processed(logger, code_file.relative_path or file_name, len(classes), len(functions))
    return FileMeta(file_name=file_name, classes=classes, functions=functions)


def parse_repository(code_files: List[CodeFile]) -> List[FileMeta]:
    """
    Parse all code files and return a list of FileMeta. Skips files that fail to parse.
    """
    result: List[FileMeta] = []
    for cf in code_files:
        meta = parse_code_file(cf)
        if meta is not None:
            result.append(meta)
    return result


def file_meta_to_dict(meta: FileMeta) -> dict[str, Any]:
    """Convert FileMeta to a JSON-serializable dict (for LLM context)."""
    return {
        "file_name": meta.file_name,
        "classes": [
            {
                "name": c.name,
                "docstring": c.docstring,
                "methods": [
                    {
                        "name": m.name,
                        "parameters": m.parameters,
                        "returns": m.returns,
                        "docstring": m.docstring,
                    }
                    for m in c.methods
                ],
            }
            for c in meta.classes
        ],
        "functions": [
            {
                "name": f.name,
                "parameters": f.parameters,
                "returns": f.returns,
                "docstring": f.docstring,
            }
            for f in meta.functions
        ],
    }
