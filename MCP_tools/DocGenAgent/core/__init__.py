"""Core components: repo reader, code parser, doc generator, markdown builder, file writer."""

from core.repo_reader import CodeFile, read_repository
from core.code_parser import FileMeta, FunctionMeta, ClassMeta, parse_repository, parse_code_file
from core.doc_generator import generate_documentation
from core.file_writer import write_readme, ensure_output_dir

__all__ = [
    "CodeFile",
    "read_repository",
    "FileMeta",
    "FunctionMeta",
    "ClassMeta",
    "parse_repository",
    "parse_code_file",
    "generate_documentation",
    "write_readme",
    "ensure_output_dir",
]
