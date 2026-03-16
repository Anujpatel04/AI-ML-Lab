"""
Write generated documentation to outputs/generated_docs/ and README.md.
"""

from pathlib import Path

from utils.config import OUTPUT_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


def ensure_output_dir() -> Path:
    """Create outputs/generated_docs if it does not exist. Return the path."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def write_readme(content: str) -> Path:
    """
    Write the full README content to outputs/generated_docs/README.md.
    Overwrites existing file. Returns the path to the written file.
    """
    ensure_output_dir()
    readme_path = OUTPUT_DIR / "README.md"
    readme_path.write_text(content, encoding="utf-8")
    logger.info("file_writer | written | path=%s", readme_path)
    return readme_path
