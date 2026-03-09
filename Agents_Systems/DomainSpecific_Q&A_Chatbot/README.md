# Domain-Specific Q&A Chatbot

A domain-focused RAG chatbot that scrapes top websites for a selected domain, stores content in Pinecone, and answers questions only from the indexed knowledge.

> Note: This project is part of the Anuj-AI-ML-Lab repository.

## What it does

- Accepts a domain (e.g., Healthcare, Finance, AI)
- Scrapes top domain‑relevant sites using Firecrawl
- Chunks, embeds, and stores content in Pinecone
- Answers questions using retrieved context only
- Clears old domain data before re‑indexing

## How to run

Set your API keys in `/Users/anuj/Desktop/Anuj-AI-ML-Lab/.env`:

```
OPENAI_API_KEY=your-api-key
FIRECRAWL_API_KEY=your-api-key
PINECONE_API_KEY=your-api-key
```

Run the Streamlit UI:

```bash
streamlit run /Users/anuj/Desktop/Anuj-AI-ML-Lab/AI_AGENTS/DomainSpecific_Q&A_Chatbot/app.py
```

Run the CLI:

```bash
python /Users/anuj/Desktop/Anuj-AI-ML-Lab/AI_AGENTS/DomainSpecific_Q&A_Chatbot/app.py --domain "AI"
```

## Example flow

1. Enter a domain (e.g., AI)
2. Click **Scrape & Index**
3. Ask questions in the chat box

## Design decisions

- Pinecone namespaces per domain
- Firecrawl for clean main‑content extraction
- Strict domain‑only answering