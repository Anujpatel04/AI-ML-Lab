#!/bin/bash
# Run script for Codebase Q&A MCP Agent

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

# Run the main script with all arguments
python main.py "$@"

