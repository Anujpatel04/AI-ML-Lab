# Music Generator Agent

> **Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab)** - A comprehensive collection of AI/ML projects, LLM applications, agents, RAG systems, and core machine learning implementations.

A Streamlit-based application that generates music using the ModelsLab Media Generation API and OpenAI GPT-4o. Users provide text prompts describing the desired music, and the application generates MP3 audio tracks.

## Features

- Music generation from text prompts using ModelsLab API
- GPT-4o powered prompt processing and optimization
- MP3 audio output with playback and download capabilities
- Streamlit web interface
- Automatic API key loading from environment variables

## Prerequisites

- Python 3.10+
- OpenAI API key
- ModelsLab API key (Media Generation API)

## Installation

```bash
cd AI_AGENTS/MusicGenrator_Agent
pip install -r requirements.txt
```

## Configuration

Add the following to your root `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
MODELSLAB_API_KEY=your_modelslab_api_key
```

API keys are automatically loaded from `.env` - no manual entry required.

## Usage

```bash
streamlit run music_generator_agent.py
```

Access at `http://localhost:8501`

1. Enter a music generation prompt (e.g., "Generate a 30 second classical music piece")
2. Click "Generate Music"
3. Play and download the generated MP3 file

## API Keys

- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **ModelsLab API Key**: Get from [ModelsLab Dashboard](https://modelslab.com/dashboard/api-keys) - Use Media Generation API

## Dependencies

- agno
- openai
- streamlit
- requests
- python-dotenv