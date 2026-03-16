#!/usr/bin/env bash
# Run the AI Documentation Generator API. Uses repo root .venv if not already in a venv.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR"
if [ -z "$VIRTUAL_ENV" ] && [ -d "$REPO_ROOT/.venv" ]; then
  source "$REPO_ROOT/.venv/bin/activate"
fi
exec uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
