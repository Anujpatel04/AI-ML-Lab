# Resume Analysis & Job Suggestions Agent

> **Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab)** - A comprehensive collection of AI/ML projects, LLM applications, agents, RAG systems, and core machine learning implementations.

AI-powered resume analysis that evaluates quality, suggests improvements, and recommends job titles. Powered by DeepSeek AI.

## Features

- Resume upload (PDF/TXT)
- Quality scoring with assessment
- Prioritized improvement suggestions
- Job recommendations with match scores
- Skills and industry identification

## Installation

```bash
cd AI_AGENTS/ResumeJOB_Suggestions
pip install -r requirements.txt
```

## Configuration

Add to root `.env`:

```env
DEEPSEEK_API_KEY=your-deepseek-api-key-here
```

## Usage

```bash
streamlit run app.py
```

Upload resume → Click "Analyze Resume" → Review results
