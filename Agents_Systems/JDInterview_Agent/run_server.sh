#!/usr/bin/env bash
# Run Adaptive AI Interview Simulator (use repo .venv if present).
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
if [ -x "$VENV_PYTHON" ]; then
  exec "$VENV_PYTHON" -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
fi
exec python3 -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
