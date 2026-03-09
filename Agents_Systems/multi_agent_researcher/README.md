## Multi-Agent AI Researcher

This Streamlit app uses a team of OpenAI-powered agents to research top HackerNews stories and users, then summarize the findings.

### Requirements
- Python 3.10+
- OpenAI API key in the root `.env` of this repo:

```bash
OPENAI_API_KEY=your_key_here
```

### Setup

From the repository root:

```bash
cd AI_AGENTS/multi_agent_researcher
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Run the app

```bash
cd AI_AGENTS/multi_agent_researcher
source .venv/bin/activate
streamlit run research_agent.py
```

Open the local URL shown in the terminal (for example `http://localhost:8502`).

### What you see
- **Team Summary**: Combined answer from the HackerNews research team.
- **HackerNews Researcher**: Stories and metadata from HackerNews.
- **Web Searcher**: Web context around the topic.
- **Article Reader**: Extracted content from linked articles.


