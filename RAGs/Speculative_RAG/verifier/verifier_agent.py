"""
Verifier agent: large model verifies the draft against retrieved context and produces final answer.
"""

import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)

VERIFIER_SYSTEM = """You are a verification assistant. You will be given:
1. The user's question
2. Retrieved reference documents (context)
3. A draft answer produced by a faster model

Your tasks:
- Check whether the draft answer is supported by the retrieved context.
- Correct any hallucinations or unsupported claims using the context.
- Improve clarity and completeness where needed.
- If the draft is already accurate and complete given the context, you may keep it with minor polish.

Respond with a JSON object containing exactly two keys:
- "feedback": A short note on what you corrected or improved (or "No changes needed" if the draft was accurate).
- "final_answer": The verified, improved final answer to the user's question.

Output only valid JSON, no other text."""


def verify_and_improve(
    client,
    model: str,
    question: str,
    context_chunks: List[str],
    draft_answer: str,
    *,
    temperature: float = 0.2,
    max_tokens: int = 1024,
) -> Tuple[str, str]:
    """
    Verify the draft answer against retrieved context and return improved answer and feedback.

    Args:
        client: OpenAI or AzureOpenAI client.
        model: Verifier model name or Azure deployment name.
        question: User question.
        context_chunks: Retrieved document chunks.
        draft_answer: Draft answer from the small model.
        temperature: Sampling temperature.
        max_tokens: Maximum tokens.

    Returns:
        (final_answer, feedback) tuple.
    """
    context = "\n\n---\n\n".join(context_chunks)
    user_content = f"""Question: {question}

Retrieved context:
{context}

Draft answer:
{draft_answer}

Provide your verification as JSON with "feedback" and "final_answer" keys."""

    messages = [
        {"role": "system", "content": VERIFIER_SYSTEM},
        {"role": "user", "content": user_content},
    ]
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = (resp.choices[0].message.content or "").strip()
        return _parse_verifier_response(text, draft_answer)
    except Exception as e:
        logger.exception("Verification failed: %s", e)
        raise


def _parse_verifier_response(text: str, fallback_answer: str) -> Tuple[str, str]:
    """Parse verifier JSON; on failure return fallback answer and error message."""
    import json
    text = text.strip()
    # Allow markdown code block
    if "```" in text:
        start = text.find("```")
        if start >= 0:
            start = text.find("\n", start) + 1
            end = text.find("```", start)
            if end > start:
                text = text[start:end]
    try:
        data = json.loads(text)
        feedback = data.get("feedback", "No feedback provided.")
        final = data.get("final_answer", fallback_answer)
        return (final.strip(), str(feedback).strip())
    except json.JSONDecodeError as e:
        logger.warning("Verifier response was not valid JSON: %s", e)
        return (fallback_answer, f"Verifier output was not valid JSON; using draft. Error: {e}")
