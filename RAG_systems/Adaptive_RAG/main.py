"""
CLI entry point for Adaptive RAG. User inputs query; system returns answer.
"""

import sys
from pathlib import Path

# Ensure project root is on path when run from repo root
_project_root = Path(__file__).resolve().parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from router import route
from utils.logger import get_logger

logger = get_logger(__name__)


def main() -> int:
    print("Adaptive RAG – enter a question (empty line to exit).")
    try:
        while True:
            try:
                line = input("\nQuery: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            answer = route(line)
            print("\nAnswer:", answer)
    except Exception as e:
        logger.exception("Pipeline failed")
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
