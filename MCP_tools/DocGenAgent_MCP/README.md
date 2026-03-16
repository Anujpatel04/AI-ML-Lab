# AI Documentation Generator Agent

Generate markdown documentation for a local Python repository via a simple FastAPI endpoint.

![AI Documentation Generator – Architecture](../DocGenAgent/docs/doc_gen_agent_architecture.svg)

## Quick Start

```bash
source /Users/anuj/Desktop/GITHUB/Anuj-AI-ML-Lab/.venv/bin/activate
cd MCP_tools/DocGenAgent_MCP
pip install -r requirements.txt
./run.sh
```

## Usage

Call the API with a **local directory path** (not a GitHub URL):

```bash
curl -X POST http://localhost:8000/generate-docs \
  -H "Content-Type: application/json" \
  -d '{"repo_path": "/absolute/path/to/your/repo"}'
```

On success, a `README.md` is written to `outputs/generated_docs/` inside this project and the response includes its path.

## Configuration

Uses the repo root `.env`:

- Azure OpenAI (default): `AZURE_ENDPOINT`, `AZURE_KEY`, `API_VERSION`
- Provider: `DOCGEN_LLM_PROVIDER` = `azure` \| `openai` \| `ollama`
- OpenAI / Ollama: `OPENAI_API_KEY`, `OLLAMA_BASE_URL`, `OLLAMA_MODEL`
