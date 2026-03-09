# Content Rewriter Agent

A CLI-based agent that rewrites content for LinkedIn, Twitter/X, or Blog with a specified tone while preserving the original meaning.

> Note: This project is part of the Anuj-AI-ML-Lab repository.

## What it does

- Accepts original content, target platform, and tone
- Applies platform-specific writing style
- Improves clarity, flow, and engagement
- Returns only the rewritten content

## How to run

Set your API key in `/Users/anuj/Desktop/Anuj-AI-ML-Lab/.env`:

```
OPENAI_API_KEY=your-api-key
OPENAI_MODEL=gpt-4o-mini

Or run interactively:

```bash
python app.py
```

## Example

Input:

- Content: "We launched a new analytics feature for small teams."
- Platform: LinkedIn
- Tone: Professional
- Constraints: "End with a CTA and keep under 120 words"

Output:

We just launched a new analytics feature built for small teams who need clear insights without complexity. It highlights key trends, saves time on reporting, and helps you make faster decisions with confidence. If you are looking to improve how your team tracks performance, this is worth a look. Want to see it in action?

## Design decisions

- Single file for portability
- Deterministic behavior with low temperature
- Environment-based configuration
