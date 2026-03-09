#!/usr/bin/env bash
# Run Context Compression RAG using repo root .venv.
set -e
SCRIPT_DIR="$(dirname "$0")"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
if [ ! -x "$VENV_PYTHON" ]; then
  echo "Repo .venv not found. Create it and install deps:"
  echo "  cd $REPO_ROOT && python3 -m venv .venv && .venv/bin/pip install -r RAGs/Context_Compression_RAG/requirements.txt"
  exit 1
fi
"$VENV_PYTHON" -m streamlit run app.py -- "$@"
