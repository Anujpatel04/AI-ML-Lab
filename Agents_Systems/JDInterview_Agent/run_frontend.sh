#!/usr/bin/env bash
# Run Streamlit frontend (uses repo .venv if present).
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
if [ -x "$VENV_PYTHON" ]; then
  exec "$VENV_PYTHON" -m streamlit run frontend.py --server.port 8501
fi
exec python3 -m streamlit run frontend.py --server.port 8501
