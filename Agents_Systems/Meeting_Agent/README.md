## AI Meeting Preparation Agent
This Streamlit application leverages multiple AI agents to create comprehensive meeting preparation materials. It uses DeepSeek API and the Serper API for web searches to generate context analysis, industry insights, meeting strategies, and executive briefings.

### Features

- Multi-agent AI system for thorough meeting preparation
- Utilizes DeepSeek API for AI-powered analysis and content generation
- Web search capability using Serper API (pre-configured)
- Generates detailed context analysis, industry insights, meeting strategies, and executive briefings

### How to get Started?

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note:** This project requires Python 3.10 or higher. Use `python3.11` if you have multiple Python versions installed.

2. Configure DeepSeek API Key

- Add your DeepSeek API key to the `.env` file in the root directory:
  ```
  DEEPSEEK_API_KEY=your_deepseek_api_key_here
  ```
- The app will automatically load the API key from the `.env` file.

3. Serper API Key

- The Serper API key is pre-configured in the application. No manual setup required.

4. Run the Streamlit App
```bash
streamlit run meeting_agent.py
```
