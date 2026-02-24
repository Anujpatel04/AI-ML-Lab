# Git Q&A MCP Agent

Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab/tree/main).

MCP (Model Context Protocol) agent for Q&A over a Git repository. Uses repository context and architecture to answer questions about the codebase.

## Requirements

- Python 3.8+
- Dependencies (see `requirements.txt` if present, or run setup script)

## Installation

1. From repo root or this folder, run the setup script:

   ```bash
   ./scripts/setup_env.sh
   ```

2. Configure any required environment variables (e.g. API keys in root `.env`).

## Usage

- Run the agent: `./scripts/run.sh` or `./scripts/run_frontend.sh` for the Streamlit frontend.
- Or activate the venv and run: `python main.py`

See [CONTRIBUTING.md](https://github.com/Anujpatel04/Anuj-AI-ML-Lab/blob/main/CONTRIBUTING.md) and the root [README](https://github.com/Anujpatel04/Anuj-AI-ML-Lab#readme) for project conventions.
