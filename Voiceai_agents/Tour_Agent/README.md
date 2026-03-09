# AI Audio Tour Agent

> **Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab)** - A comprehensive collection of AI/ML projects, LLM applications, agents, RAG systems, and core machine learning implementations.

A conversational voice agent system that generates immersive, self-guided audio tours based on user location, areas of interest, and tour duration. Built on a multi-agent architecture using OpenAI Agents SDK, real-time information retrieval, and expressive TTS for natural speech output.

## Features

### Multi-Agent Architecture

- **Orchestrator Agent**: Coordinates the overall tour flow, manages transitions, and assembles content from all expert agents
- **History Agent**: Delivers insightful historical narratives with an authoritative voice
- **Architecture Agent**: Highlights architectural details, styles, and design elements using a descriptive and technical tone
- **Culture Agent**: Explores local customs, traditions, and artistic heritage with an enthusiastic voice
- **Culinary Agent**: Describes iconic dishes and food culture in a passionate and engaging tone

### Location-Aware Content Generation

- Dynamic content generation based on user-input location
- Real-time web search integration to fetch relevant, up-to-date details
- Personalized content delivery filtered by user interest categories

### Customizable Tour Duration

- Selectable tour length: 5-60 minutes
- Time allocations adapt to user interest weights and location relevance
- Ensures well-paced and proportioned narratives across sections

### Expressive Speech Output

- High-quality audio generated using OpenAI TTS (tts-1 model)
- Natural, conversational voice output
- MP3 format for easy playback and download

## Prerequisites

- Python 3.10+
- OpenAI API key

## Installation

1. Navigate to the project directory:
```bash
cd VOICE_AGENTS/Tour_Agent
```

2. Create a virtual environment:
```bash
python3 -m venv venv
```

3. Activate the virtual environment:
```bash
source venv/bin/activate
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Configuration

Add your OpenAI API key to the root `.env` file:

```env
OPENAI_API_KEY=your_openai_api_key
```

The application will automatically load the API key from the root `.env` file. No manual entry required.

## Usage

### Option 1: Using the run script (Recommended)

```bash
bash run.sh
```

### Option 2: Manual activation

```bash
source venv/bin/activate
streamlit run tour_agent.py
```

The app will be available at `http://localhost:8501` or `http://localhost:8502`

### Using the Application

1. Enter a location (e.g., "Paris", "New York", "Tokyo")
2. Select your interests (History, Architecture, Culinary, Culture)
3. Choose tour duration (5-60 minutes)
4. Select voice style
5. Click "Generate Tour"
6. Wait for the tour to be generated
7. Listen to and download the generated MP3 audio tour

## API Keys

- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)

## Dependencies

- openai
- openai-agents
- streamlit
- pydantic
- python-dotenv
- rich

## License

Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab) - MIT License
