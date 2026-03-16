# AI Documentation Generator Agent

A production-quality service that analyzes a code repository and generates professional documentation using an LLM. The system is modular, maintainable, and designed for future integration with GitHub webhooks or CI/CD pipelines.

## Project Purpose

- **Read** a local repository and identify code files (`.py`).
- **Extract** classes, functions, method signatures, parameters, return annotations, and docstrings via Python AST.
- **Generate** documentation (descriptions, parameters, returns, usage examples) using an LLM.
- **Produce** structured markdown and a complete `README.md` in `outputs/generated_docs/`.

## Architecture

```
project_root/
├── api/
│   └── main.py              # FastAPI app, POST /generate-docs
├── core/
│   ├── repo_reader.py      # Traverse repo, return CodeFile list
│   ├── code_parser.py      # AST extraction → FileMeta (classes, functions)
│   ├── doc_generator.py    # Orchestrates read → parse → LLM → markdown → write
│   ├── markdown_builder.py # Sections: Overview, TOC, Installation, Usage, API Reference
│   └── file_writer.py      # Write README to outputs/generated_docs/
├── llm/
│   ├── chains.py           # LangChain + Azure OpenAI / OpenAI / Ollama
│   └── prompts.py          # Prompt templates for docs
├── utils/
│   ├── config.py           # Load .env (Azure, OpenAI, Ollama), paths
│   └── logger.py           # Structured logging
├── outputs/
│   └── generated_docs/     # Generated README.md and future artifacts
├── requirements.txt
└── README.md
```

## Configuration

Configuration is loaded from the **repository root** `.env` (e.g. `Anuj-AI-ML-Lab/.env`).

| Variable | Description |
|----------|-------------|
| `AZURE_ENDPOINT` or `AZURE_OPENAI_ENDPOINT` | Azure OpenAI endpoint URL (e.g. `https://xxx.openai.azure.com/openai/deployments/gpt-4o/...`) |
| `AZURE_KEY` or `AZURE_OPENAI_API_KEY` | Azure API key |
| `API_VERSION` or `AZURE_OPENAI_API_VERSION` | Azure API version |
| `DOCGEN_LLM_PROVIDER` | `azure` (default), `openai`, or `ollama` |
| `OPENAI_API_KEY` | Used when `DOCGEN_LLM_PROVIDER=openai` |
| `OLLAMA_BASE_URL` | Used when `DOCGEN_LLM_PROVIDER=ollama` (default `http://localhost:11434`) |
| `OLLAMA_MODEL` | Model name for Ollama |

## How to Run

1. **Use the repo root virtual environment and install dependencies**

   ```bash
   source /Users/anuj/Desktop/GITHUB/Anuj-AI-ML-Lab/.venv/bin/activate
   cd MCP_tools/DocGenAgent_MCP
   pip install -r requirements.txt
   ```

2. **Ensure `.env`** at the repo root contains Azure OpenAI (or OpenAI/Ollama) settings.

3. **Start the API**

   From DocGenAgent_MCP (with repo `.venv` activated):

   ```bash
   cd MCP_tools/DocGenAgent_MCP
   PYTHONPATH=. uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```

   Or use the run script (it auto-activates repo `.venv` if present):

   ```bash
   ./MCP_tools/DocGenAgent_MCP/run.sh
   ```

4. **Generate docs**

   ```bash
   curl -X POST http://localhost:8000/generate-docs \
     -H "Content-Type: application/json" \
     -d '{"repo_path": "./sample_project"}'
   ```

   Or use an absolute path:

   ```json
   { "repo_path": "/Users/anuj/Desktop/GITHUB/Anuj-AI-ML-Lab/MCP_tools/DocGenAgent" }
   ```

## Example API Request / Response

**Request**

```http
POST /generate-docs
Content-Type: application/json

{
  "repo_path": "./sample_project"
}
```

**Response (success)**

```json
{
  "status": "success",
  "readme_path": "/path/to/MCP_tools/DocGenAgent/outputs/generated_docs/README.md",
  "message": null
}
```

**Response (error)**

```json
{
  "status": "error",
  "readme_path": null,
  "message": "Repository path does not exist: ./missing"
}
```

## Logging

Structured logs include:

- `repo_scanned` – repository path and file count
- `files_processed` – file path, class count, function count
- `llm_call` – provider/prompt type
- `documentation_generated` – output path

Log level is controlled by `LOG_LEVEL` in `.env` (e.g. `INFO`).

## Engineering Notes

- **Modular design**: Repo reader, parser, LLM chains, markdown builder, and file writer are separate modules.
- **Type hints and docstrings** used across the codebase.
- **No hardcoded paths**: Output directory and repo root come from config and parameters.
- **Clean error handling**: Invalid paths return 4xx; pipeline errors return a response with `status: "error"` and message.
