#!/usr/bin/env bash
# Run Multi-Agent RAG Streamlit app using repo root .venv.
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
  echo "Repo .venv not found. Create and install:"
  echo "  cd $REPO_ROOT && python3 -m venv .venv && .venv/bin/pip install -r RAG_systems/MultiAgent_RAG/requirements.txt"
  exit 1
fi
"$VENV_PYTHON" -m streamlit run frontend/app.py -- "$@"
