# AI Journalist Agent

An AI-powered journalist agent that generates high-quality articles using DeepSeek API. This Streamlit app automates the process of researching, writing, and editing articles, allowing you to create compelling content on any topic.

## Features

- **Web Search**: Automatically searches the web for relevant information on any given topic
- **Article Writing**: Writes well-structured, informative, and engaging articles based on research
- **Content Editing**: Refines the generated content to meet high editorial standards

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

**DeepSeek API Key:**
- Add your DeepSeek API key to the `.env` file in the root directory:
  ```
  DEEPSEEK_API_KEY=your_deepseek_api_key_here
  ```
- The app will automatically load the API key from the `.env` file.
- Get your API key from [DeepSeek Platform](https://platform.deepseek.com/)

**SerpAPI Key:**
- The SerpAPI key is pre-configured in the application, so no manual setup is required.

### 3. Run the App

```bash
streamlit run journalist_agent.py
```

## How It Works

The AI Journalist Agent uses a three-agent workflow:

1. **Searcher**: Generates search terms based on the topic and searches the web for relevant URLs using SerpAPI. Returns the most relevant sources.

2. **Writer**: Retrieves article text from the provided URLs using Newspaper4k and writes a comprehensive, NYT-worthy article based on the extracted information.

3. **Editor**: Performs final editing and refinement of the generated article to ensure it meets high editorial standards.

Simply enter a topic, and the agent will handle the research, writing, and editing process automatically.

## Repository

This project is part of the [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab/tree/main/Single_AI_Agents/Journalist_Agent) repository.

