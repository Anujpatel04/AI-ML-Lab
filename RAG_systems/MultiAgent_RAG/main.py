"""
CLI entry point for Multi-Agent RAG.
"""

import sys
from pathlib import Path

# Ensure project root is on path
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from core.orchestrator import Orchestrator, PipelineResult


def main() -> int:
    orchestrator = Orchestrator()
    print("Multi-Agent RAG – enter a question (empty line to exit).")
    try:
        while True:
            try:
                line = input("\nQuestion: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            result: PipelineResult = orchestrator.run(line)
            print("\n--- Retrieved docs ---")
            for i, doc in enumerate(result.retrieved_docs[:3], 1):
                print(f"{i}. [{doc.get('score', 0):.3f}] {doc.get('content', '')[:200]}...")
            print("\n--- Answer ---")
            print(result.final_answer)
            print("\n--- Verification ---", "PASS" if result.verified else "REFINED/FAIL", result.verification_reason)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
