# Chat with Tarots

An AI-powered tarot reading application that combines local language models with traditional tarot card interpretations to provide personalized readings based on natural language queries.

## Overview

This application uses a local AI model (phi4 via Ollama) to analyze tarot card draws and provide contextual interpretations. Users can ask questions in natural language, and the system will draw cards, analyze their meanings, and generate insights based on traditional tarot symbolism.

## Features

- Natural language input processing
- Local AI model integration using Ollama (phi4)
- CSV-based knowledge base containing 78 tarot cards with upright, reversed, and symbolism meanings
- Interactive Streamlit interface
- Configurable card spreads (3, 5, or 7 cards)
- Context-aware interpretations combining card meanings with user queries

## How It Works

The application loads tarot card data from a CSV file containing meanings for all 78 cards. When a user submits a query, the system:

1. Randomly draws the selected number of cards (with random reversal)
2. Retrieves the meanings and symbolism for each drawn card
3. Uses the phi4 model through LangChain to generate an interpretation
4. Combines traditional tarot meanings with the user's context to provide personalized insights

The AI model is prompted with card details, user context, and symbolism information to create coherent and meaningful readings.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Ollama installed and running locally

Install Ollama from [ollama.com](https://ollama.com/) and pull the required model:

```bash
ollama pull phi4
ollama serve
```

## Installation

1. Navigate to the project directory:

```bash
cd All_LargeLangugage_Models/chat-with-tarots
```

2. Create and activate a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will be available at `http://localhost:8501` (or the next available port).

### Using the Application

1. Select the number of cards for your spread (3, 5, or 7)
2. Enter your question or context in the text area
3. Click "Light your path: Draw and Analyze the Cards"
4. View the drawn cards and read the AI-generated interpretation

The system will display the cards (with reversed cards rotated 180 degrees) and provide a detailed analysis that combines traditional tarot meanings with your specific question.

## Project Structure

- `app.py` - Main Streamlit application
- `helpers/help_func.py` - Helper functions for card drawing and LangChain setup
- `data/tarots.csv` - Tarot card meanings database
- `images/` - Tarot card images (78 cards)
- `requirements.txt` - Python dependencies

## Technical Details

- **Framework**: Streamlit for the web interface
- **AI Model**: phi4 via Ollama (local inference)
- **LangChain**: Used for prompt templating and chain construction
- **Data Format**: CSV with semicolon delimiters, Latin-1 encoding

## Repository

This project is part of the [Anuj AI/ML Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab) collection. For more AI/ML projects and implementations, visit the main repository.

## License

This project is part of the Anuj AI/ML Lab repository and is licensed under the MIT License.
