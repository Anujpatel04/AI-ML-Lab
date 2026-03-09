# Web Scraping AI Agent

An AI-powered web scraping tool that extracts structured data from websites using natural language prompts. This agent uses ScrapeGraphAI to intelligently scrape and extract information from web pages.

## What's Inside

This folder contains two implementations:

1. **DeepSek_scrapper.py** - Uses DeepSeek API for cloud-based scraping
2. **ollama_scrapper.py** - Uses local Ollama models for offline scraping

## Prerequisites

- Python 3.9 or higher
- Streamlit
- For ai_scrapper.py: DeepSeek API key in your .env file
- For local_ai_scrapper.py: Ollama installed and running locally with llama3.1:latest model

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Playwright browsers (required for scraping):
```bash
playwright install chromium
```

## Configuration

### For DeepSek_scrapper.py (DeepSeek API)

1. Add your DeepSeek API key to the root .env file:
```
DEEPSEEK_API_KEY=your-deepseek-api-key-here
```

2. The app will automatically load the API key from the .env file.

### For ollama_scrapper.py (Ollama)

1. Make sure Ollama is installed and running on your machine
2. Pull the required model:
```bash
ollama pull llama3.1:latest
```

## Usage

### Run with DeepSeek API

```bash
streamlit run DeepSek_scrapper.py
```

This will start a Streamlit app where you can:
- Select between deepseek-chat and deepseek-reasoner models
- Enter a website URL to scrape
- Provide a natural language prompt describing what to extract
- Get structured results from the scraping operation

### Run with Local Ollama

```bash
streamlit run ollama_scrapper.py
```

This will start a Streamlit app using your local Ollama installation:
- Uses llama3.1:latest model
- Works completely offline
- No API costs

## How It Works

1. Enter the URL of the website you want to scrape
2. Provide a prompt describing what information you want to extract
3. Click the Scrape button
4. The AI agent analyzes the webpage and extracts the requested information
5. Results are displayed in structured format

## Use Cases

- Extract product information from e-commerce sites
- Gather contact details from business websites
- Collect article metadata and content
- Monitor competitor pricing and features
- Extract structured data from any website

## Requirements

See requirements.txt for the full list of dependencies. Main packages include:
- streamlit
- scrapegraphai
- playwright
- python-dotenv

## Notes

- The DeepSeek version requires an active internet connection and API key
- The local version requires Ollama to be running and the model downloaded
- Playwright browsers are needed for both versions to render JavaScript-heavy websites
- Scraping results depend on website structure and may require prompt refinement
