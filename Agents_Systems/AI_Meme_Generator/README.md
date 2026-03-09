# AI Meme Generator Agent

An AI-powered browser automation tool that generates memes using natural language prompts. The agent automatically navigates imgflip.com, selects appropriate meme templates, and creates custom memes based on your descriptions.

## Features

- Multi-LLM support (DeepSeek, Claude, OpenAI)
- Automated browser interaction with imgflip.com
- Intelligent meme template selection
- Automatic text caption generation
- Direct meme preview and download links

## Prerequisites

- Python 3.11 or higher (required for browser-use package)
- DeepSeek API key (or other LLM API keys)
- Playwright browsers installed

## Installation

1. Navigate to the project directory:
```bash
cd Single_AI_Agents/AI_Meme_Generator
```

2. Install Python dependencies:
```bash
python3.11 -m pip install -r requirements.txt
```

3. Install Playwright browsers (required for browser automation):
```bash
python3.11 -m playwright install --with-deps
```

## Configuration

### DeepSeek API Key (Default)

The app is configured to use DeepSeek API key from the root `.env` file:
- Location: `/Users/anuj/Desktop/Anuj-AI-ML-Lab/.env`
- Variable name: `DEEPSEEK_API_KEY`

If the API key is found in the `.env` file, it will be used automatically. Otherwise, you can enter it manually in the sidebar.

### Other Models

You can also use Claude or OpenAI by:
1. Selecting the model from the dropdown in the sidebar
2. Entering the corresponding API key

## Usage

1. Start the Streamlit app:
```bash
python3.11 -m streamlit run ai_meme_generator_agent.py
```

2. Open your browser to the URL shown (typically http://localhost:8501)

3. Select your preferred AI model (DeepSeek is default)

4. Enter a meme idea in the text input field

5. Click "Generate Meme" and wait for the agent to create your meme

6. View the generated meme and use the provided link to download or share

## How It Works

1. The agent receives your meme description
2. It navigates to imgflip.com/memetemplates
3. Searches for relevant meme templates based on action verbs in your prompt
4. Selects an appropriate template
5. Generates top and bottom text captions
6. Creates the meme and extracts the image URL
7. Displays the result in the app

## Requirements

See `requirements.txt` for the full list of dependencies. Main packages include:
- streamlit
- browser-use (requires Python 3.11+)
- playwright
- langchain-openai
- langchain-anthropic
- python-dotenv

## Troubleshooting

### Playwright Browser Not Found

If you see errors about missing Chromium:
```bash
python3.11 -m playwright install --with-deps
```

### Python Version Error

Make sure you're using Python 3.11 or higher:
```bash
python3.11 --version
```

### API Key Issues

- For DeepSeek: Ensure `DEEPSEEK_API_KEY` is set in your root `.env` file
- For other models: Enter the API key manually in the sidebar

## Notes

- The agent uses browser automation, so the first generation may take longer
- Internet connection is required for the agent to access imgflip.com
- The agent will retry up to 25 times if it encounters errors
- Generated memes are hosted on imgflip.com
