"""
Repository reader: traverses a repo directory and returns code files and contents.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List

from utils.logger import get_logger, log_repo_scanned

logger = get_logger(__name__)

# Supported code extensions (only .py per spec; easy to extend)
CODE_EXTENSIONS = {".py"}


@dataclass
class CodeFile:
    """A single code file with path and raw content."""

    path: str
    content: str
    relative_path: str = ""

    def __post_init__(self) -> None:
        if not self.relative_path and self.path:
            self.relative_path = str(Path(self.path).name)


def read_repository(repo_path: str) -> List[CodeFile]:
    """
    Traverse a repository directory, identify code files (.py), and return their paths and contents.

    Args:
        repo_path: Absolute or relative path to the repository root.

    Returns:
        List of CodeFile objects with path and content. Files that cannot be read are skipped.

    Raises:
        FileNotFoundError: If repo_path does not exist.
        NotADirectoryError: If repo_path is not a directory.
    """
    root = Path(repo_path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Repository path does not exist: {repo_path}")
    if not root.is_dir():
        raise NotADirectoryError(f"Repository path is not a directory: {repo_path}")

    files: List[CodeFile] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in CODE_EXTENSIONS:
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
            rel = path.relative_to(root)
            files.append(
                CodeFile(
                    path=str(path),
                    content=content,
                    relative_path=str(rel),
                )
            )
        except OSError as e:
            logger.warning("repo_reader | skip_file | path=%s | error=%s", path, e)

    log_repo_scanned(logger, repo_path, len(files))
    return files
