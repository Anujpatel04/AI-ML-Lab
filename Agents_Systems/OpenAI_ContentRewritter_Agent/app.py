import argparse
import os
import sys
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


ROOT_ENV_PATH = "/Users/anuj/Desktop/Anuj-AI-ML-Lab/.env"
DEFAULT_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a professional content rewriting AI agent.

Your task is to rewrite the provided text for a specific platform and tone while preserving the original meaning.

INPUTS YOU WILL RECEIVE:
- Original content
- Target platform (LinkedIn, Twitter/X, Blog)
- Desired tone (Professional, Casual, Conversational, Persuasive)
- Optional constraints (word limit, hashtags, emojis, CTA)

RULES:
1. Do NOT change the core message or facts.
2. Improve clarity, flow, and engagement.
3. Adapt writing style strictly to the target platform.
4. Keep language simple, confident, and natural.
5. Avoid filler, clichés, and overly generic phrases.
6. Do not mention that you are an AI or that the text was rewritten.

PLATFORM GUIDELINES:
- LinkedIn:
  - Professional, insightful, value-driven
  - Short paragraphs (1–2 lines)
  - Optional bullet points
  - Light emojis allowed (max 3)
  - End with a thoughtful question or CTA

- Twitter/X:
  - Concise and punchy
  - Maximize impact per sentence
  - Use line breaks
  - Emojis optional (max 2)
  - Hashtags optional (max 2)

- Blog:
  - Structured and detailed
  - Clear introduction, body, and conclusion
  - No emojis unless explicitly requested
  - Maintain readability and flow

OUTPUT FORMAT:
Return ONLY the rewritten content.
Do NOT add explanations, headers, or commentary.
"""


def load_env() -> None:
    load_dotenv(ROOT_ENV_PATH, override=True)


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key)


def get_model_name() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def _read_input(prompt: str) -> str:
    return input(prompt).strip()


def build_user_message(
    content: str,
    platform: str,
    tone: str,
    constraints: Optional[str],
) -> str:
    lines = [
        f"Original content:\n{content}",
        f"Target platform: {platform}",
        f"Desired tone: {tone}",
    ]
    if constraints:
        lines.append(f"Optional constraints: {constraints}")
    return "\n\n".join(lines)


def rewrite_content(
    content: str,
    platform: str,
    tone: str,
    constraints: Optional[str],
) -> str:
    if not content:
        raise ValueError("Original content is required.")
    if not platform:
        raise ValueError("Target platform is required.")
    if not tone:
        raise ValueError("Desired tone is required.")

    client = get_client()
    model = get_model_name()
    user_message = build_user_message(content, platform, tone, constraints)

    response = client.chat.completions.create(
        model=model,
        temperature=0.4,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return (response.choices[0].message.content or "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Content Rewriter Agent")
    parser.add_argument("--content", type=str, help="Original content to rewrite.")
    parser.add_argument(
        "--platform",
        type=str,
        choices=["LinkedIn", "Twitter/X", "Blog"],
        help="Target platform.",
    )
    parser.add_argument(
        "--tone",
        type=str,
        choices=["Professional", "Casual", "Conversational", "Persuasive"],
        help="Desired tone.",
    )
    parser.add_argument(
        "--constraints",
        type=str,
        help="Optional constraints (word limit, hashtags, emojis, CTA).",
    )
    return parser.parse_args()


def run_streamlit() -> None:
    st.set_page_config(page_title="Content Rewriter Agent", layout="centered")
    st.title("Content Rewriter Agent")
    st.write("Rewrite content for LinkedIn, Twitter/X, or Blog with a selected tone.")

    load_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY is not set in the environment.")
        st.info(f"Expected .env location: {ROOT_ENV_PATH}")
        st.stop()

    content = st.text_area(
        "Original content",
        placeholder="Paste the content you want to rewrite...",
        height=180,
    )
    platform = st.selectbox(
        "Target platform",
        ["LinkedIn", "Twitter/X", "Blog"],
    )
    tone = st.selectbox(
        "Desired tone",
        ["Professional", "Casual", "Conversational", "Persuasive"],
    )
    constraints = st.text_input(
        "Optional constraints",
        placeholder="Word limit, hashtags, emojis, CTA (optional)",
    )

    if "last_output" not in st.session_state:
        st.session_state.last_output = None

    if st.button("Rewrite"):
        try:
            st.session_state.last_output = rewrite_content(
                content=content.strip(),
                platform=platform,
                tone=tone,
                constraints=constraints.strip() if constraints else None,
            )
        except Exception as exc:
            st.session_state.last_output = f"Error: {exc}"

    if st.session_state.last_output:
        if st.session_state.last_output.startswith("Error:"):
            st.error(st.session_state.last_output)
        else:
            st.subheader("Rewritten Content")
            st.write(st.session_state.last_output)


def run_cli() -> None:
    load_env()
    args = parse_args()

    content = args.content or _read_input("Original content: ")
    platform = args.platform or _read_input("Target platform (LinkedIn, Twitter/X, Blog): ")
    tone = args.tone or _read_input(
        "Desired tone (Professional, Casual, Conversational, Persuasive): "
    )
    constraints = args.constraints or _read_input(
        "Optional constraints (press Enter to skip): "
    )
    constraints = constraints if constraints else None

    result = rewrite_content(content, platform, tone, constraints)
    print(result)


def is_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime import exists

        return exists()
    except Exception:
        return bool(getattr(st, "_is_running_with_streamlit", False))


if __name__ == "__main__":
    try:
        if is_streamlit_runtime():
            run_streamlit()
        else:
            run_cli()
    except KeyboardInterrupt:
        sys.exit(130)
