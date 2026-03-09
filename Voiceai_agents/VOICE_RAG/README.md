# Voice RAG Agent

> **Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab)** - A comprehensive collection of AI/ML projects, LLM applications, agents, RAG systems, and core machine learning implementations.

A voice-enabled Retrieval-Augmented Generation (RAG) system that allows users to upload PDF documents, ask questions, and receive both text and voice responses using OpenAI's SDK and text-to-speech capabilities.

## Features

- **Document Processing**: Upload and process PDF documents with automatic chunking and embedding
- **Vector Search**: Uses Qdrant vector database for efficient similarity search
- **Intelligent Querying**: Multi-agent architecture for generating clear, conversational responses
- **Voice Responses**: Real-time text-to-speech with multiple voice options
- **Audio Download**: Download generated audio responses as MP3 files
- **Professional Interface**: Clean, modern Streamlit interface for seamless interaction

## Architecture

### Document Processing
- PDF documents are split into chunks using LangChain's RecursiveCharacterTextSplitter
- Each chunk is embedded using FastEmbed
- Embeddings are stored in Qdrant for efficient retrieval

### Query Processing
1. User questions are converted to embeddings
2. Similar document chunks are retrieved from Qdrant
3. A processing agent generates clear, spoken-word friendly responses
4. A TTS agent optimizes responses for speech synthesis

### Voice Generation
- Text responses are converted to speech using OpenAI's TTS API
- Multiple voice options available (alloy, ash, ballad, coral, echo, fable, onyx, nova, sage, shimmer, verse)
- Audio can be played directly or downloaded as MP3

## Prerequisites

- Python 3.11+ (Python 3.14 is not compatible with fastembed)
- OpenAI API key
- Qdrant Cloud account (or self-hosted Qdrant instance)

## Installation

1. Navigate to the project directory:
```bash
cd VOICE_AGENTS/VOICE_RAG
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

Add your API keys to the root `.env` file:

```env
OPENAI_API_KEY=your-openai-api-key
QDRANT_URL=your-qdrant-url
QDRANT_API_KEY=your-qdrant-api-key
```

The application will automatically load API keys from the root `.env` file. No manual configuration required.

## Usage

### Option 1: Using the virtual environment

```bash
source venv/bin/activate
streamlit run rag_voice.py
```

### Option 2: Direct execution

```bash
./venv/bin/streamlit run rag_voice.py
```

The app will be available at `http://localhost:8501` or `http://localhost:8502`

### Using the Application

1. **Upload Document**: Click "Upload PDF Document" and select a PDF file
2. **Wait for Processing**: The document will be processed and embedded automatically
3. **Ask Questions**: Enter your question in the query interface
4. **Receive Response**: Get both text and audio responses
5. **Download Audio**: Download the audio response as an MP3 file

## API Keys

- **OpenAI API Key**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
- **Qdrant URL & API Key**: Get from [Qdrant Cloud](https://cloud.qdrant.io/) or use self-hosted instance

## Dependencies

- openai-agents
- streamlit
- qdrant-client
- fastembed
- langchain
- langchain-community
- langchain-openai
- langchain-text-splitters
- openai
- pypdf
- python-dotenv
- sounddevice
