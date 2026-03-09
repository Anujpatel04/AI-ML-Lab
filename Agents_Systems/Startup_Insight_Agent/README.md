# AI Startup Insight Agent

A web extraction and analysis tool that extracts structured data from startup websites using Firecrawl's FIRE-1 agent and provides AI-powered business analysis with DeepSeek AI.

## Features

- Extract structured data from company websites
- AI-powered business analysis and insights
- Process multiple websites simultaneously
- Interactive web interface

## Requirements

- Python 3.10 or higher (Python 3.11 recommended)
- DeepSeek API key
- Internet connection

## Installation

1. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

   Or with Python 3.11:
   ```bash
   python3.11 -m pip install -r requirements.txt
   ```

2. **Configure DeepSeek API Key**

   Add your DeepSeek API key to the `.env` file in the root directory:

   ```
   DEEPSEEK_API_KEY=your_deepseek_api_key_here
   ```

   The Firecrawl API key is pre-configured. No additional setup needed.

## Usage

1. **Run the Application**

   ```bash
   streamlit run ai_startup_insight_fire1_agent.py
   ```

   Or with Python 3.11:
   ```bash
   python3.11 -m streamlit run ai_startup_insight_fire1_agent.py
   ```

2. **Use the Application**

   - Enter one or more website URLs (one per line) in the text area
   - Click "Start Analysis" to begin extraction
   - View extracted data and AI analysis in the tabbed interface

## Example Websites

- https://www.spurtest.com
- https://cluely.com
- https://www.harvey.ai

## Technologies

- Firecrawl FIRE-1: Web extraction agent
- Agno Agent Framework: AI analysis
- DeepSeek AI: Business insight generation
- Streamlit: Web interface
