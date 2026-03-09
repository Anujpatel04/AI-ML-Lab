# AI Voice Agent

> **Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab)** - A comprehensive collection of AI/ML projects, LLM applications, agents, RAG systems, and core machine learning implementations.

A self-hosted AI voice agent for real-time voice conversations using Deepgram (speech-to-text) and OpenAI GPT-3.5-turbo (responses with TTS). Suitable for sales calls, customer support, and voice interactions.

## Features

- Real-time voice transcription (Deepgram Nova-2)
- GPT-3.5-turbo conversational AI
- Text-to-speech responses
- Conversation memory
- Automatic microphone muting during responses

## Prerequisites

- Python 3.11+
- Deepgram API key
- OpenAI API key
- Microphone access permissions

## Installation

```bash
cd AI_AGENTS/AI-Voice-Agent
python3 -m venv myenv
source myenv/bin/activate
brew install portaudio  # macOS only
pip install -r requirements.txt
```

## Configuration

Add API keys to the root `.env` file:

```env
DEEPGRAM_API_KEY=your-deepgram-api-key
OPENAI_API_KEY=your-openai-api-key
```

API keys are automatically loaded from the root `.env` file.

## Usage

```bash
source myenv/bin/activate
python app.py
```

Grant microphone permissions when prompted, then speak into your microphone. Press Enter to stop.

## API Keys

- **Deepgram**: [Deepgram Console](https://console.deepgram.com/)
- **OpenAI**: [OpenAI Platform](https://platform.openai.com/api-keys)

## How It Works

1. Microphone captures audio → Deepgram transcribes to text
2. GPT-3.5-turbo generates contextual response
3. OpenAI TTS converts response to speech
4. Audio played through speakers
