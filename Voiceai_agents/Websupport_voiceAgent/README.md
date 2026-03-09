# Customer Support Voice Agent

> **Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab)** - A comprehensive collection of AI/ML projects, LLM applications, agents, RAG systems, and core machine learning implementations.

A voice-powered customer support agent that answers questions about documentation using OpenAI GPT-4o and TTS. The system crawls documentation websites, stores content in a Qdrant vector database, and provides text and voice responses.

## Features

- Automated documentation crawling using Firecrawl
- Semantic search with FastEmbed embeddings and Qdrant vector database
- AI-powered response generation using GPT-4o
- Text-to-speech conversion with OpenAI TTS API
- Multiple voice options (alloy, ash, ballad, coral, echo, fable, onyx, nova, sage, shimmer, verse)
- Streamlit web interface with automatic API key loading from `.env`

## Prerequisites

- Python 3.10+
- OpenAI API key
- Qdrant Cloud account or self-hosted instance
- Firecrawl API key

## Installation

```bash
cd VOICE_AGENTS/customer_support_voice_agent
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root with:

```env
OPENAI_API_KEY=your_openai_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

API keys are automatically loaded from `.env` - no manual entry required.

## Usage

```bash
streamlit run customer_support_voice_agent.py
```

Access at `http://localhost:8501`

1. Enter documentation URL
2. Select voice preference
3. Click "Initialize System" to crawl and process documentation
4. Ask questions and receive text and voice responses

## Architecture

- **Firecrawl**: Documentation web scraping
- **Qdrant**: Vector database for semantic search
- **FastEmbed**: Embedding generation
- **GPT-4o**: Response generation and text optimization
- **OpenAI TTS**: Speech synthesis

## Dependencies

- openai
- firecrawl-py
- qdrant-client
- fastembed
- streamlit
- openai-agents
- python-dotenv

## License

Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab) - MIT License
