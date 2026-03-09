## Personal Context Memory Agent

Production-ready Streamlit app that stores high-signal user context in
`memory.json` and injects relevant memory into responses. Optimized for
drafting message and email replies. This agent is part of the Anuj AI ML Lab.

### Features
- Local JSON memory store for preferences, writing style, decisions
- Relevance-based context injection to keep prompts concise
- Manual run flow with memory viewer and clear-memory confirmation
- Built-in fallback responses for common messaging scenarios

### Quickstart
```
pip install -r requirements.txt
streamlit run app.py
```

### Configuration
- Replace `call_llm()` in `app.py` with your preferred LLM provider.
- Memory persists in `memory.json` in the same folder as `app.py`.

### Usage
- Enter a message or question in the input box and click `Run Agent`.
- Ask for help drafting replies to emails or messages.
- Review saved preferences, style, and decisions in the memory viewer.
- Use `Clear Memory` to reset stored context.
