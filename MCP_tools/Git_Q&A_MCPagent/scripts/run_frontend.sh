#!/bin/bash
# Run Streamlit frontend for Codebase Q&A MCP Agent

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
else
    echo "Error: Virtual environment not found!"
    echo "Please run: ./setup_env.sh first"
    exit 1
fi

# Run Streamlit app
streamlit run app.py --server.port 8501 --server.address localhost

