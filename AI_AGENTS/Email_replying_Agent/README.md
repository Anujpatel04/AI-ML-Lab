# Email Clarity Rewriter Agent

Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab/tree/main).

Streamlit app that rewrites and clarifies emails using DeepSeek. Choose tone (Professional, Polite, Assertive, Concise), optionally shorten, highlight unclear sentences, and improve structure. Output includes rewritten email, improvements summary, clarity issues, and a basic readability score.

## Requirements

- Python 3.9+
- `DEEPSEEK_API_KEY` in the project root `.env` (parent of `AI_AGENTS`)

## Installation

```bash
cd AI_AGENTS/Email_replying_Agent
pip install -r requirements.txt
```

## Usage

```bash
streamlit run Email_Clarity_Rewriter_Agent.py
```

## Configuration

- **DEEPSEEK_API_KEY:** Required.
- **DEEPSEEK_MODEL:** Optional; default `deepseek-chat`.
- **DEEPSEEK_BASE_URL:** Optional; default `https://api.deepseek.com/v1`.
