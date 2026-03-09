# Browser MCP Agent

A Streamlit application that allows you to browse and interact with websites using natural language commands through the Model Context Protocol (MCP) and MCP-Agent with Playwright integration.

## Features

- Natural Language Interface: Control a browser with simple English commands
- Full Browser Navigation: Visit websites and navigate through pages
- Interactive Elements: Click buttons, fill forms, and scroll through content
- Visual Feedback: Take screenshots of webpage elements
- Information Extraction: Extract and summarize content from webpages
- Multi-step Tasks: Complete complex browsing sequences through conversation

## Requirements

- Python 3.11+ (required for mcp-agent)
- Node.js and npm (for Playwright)
- Azure OpenAI API key

## Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Verify Node.js is installed:
   ```bash
   node --version
   npm --version
   ```

3. Configure Azure OpenAI:
   - Copy `mcp_agent.secrets.yaml.example` to `mcp_agent.secrets.yaml`
   - Add your Azure OpenAI API key to `mcp_agent.secrets.yaml`
   - Update `mcp_agent.config.yaml` with your Azure endpoint and deployment name

## Configuration

Edit `mcp_agent.config.yaml`:
```yaml
openai:
  default_model: "gpt-4o"
  base_url: "https://your-resource.openai.azure.com/openai/deployments/your-deployment"
  api_version: "2025-01-01-preview"
```

Edit `mcp_agent.secrets.yaml`:
```yaml
openai:
  api_key: "your-azure-openai-api-key"
```

## Running the App

Start the Streamlit app:
```bash
python3.11 -m streamlit run main.py
```

The app will be available at http://localhost:8501

## Usage

Enter natural language commands in the text area and click "Run Command". Examples:

- "Go to google.com"
- "Navigate to github.com and search for Python"
- "Click on the login button"
- "Scroll down and take a screenshot"
- "Extract all links from this page"

## Architecture

- Streamlit for the user interface
- MCP (Model Context Protocol) to connect the LLM with tools
- Playwright for browser automation
- MCP-Agent for the Agentic Framework
- Azure OpenAI for command interpretation and response generation

## Notes

- The app uses Azure OpenAI with a monkey-patch to handle the api-version query parameter
- Max tokens is set to 4096 to comply with Azure OpenAI limits
- Browser tools are automatically available through the Playwright MCP server
