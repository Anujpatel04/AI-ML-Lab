#!/usr/bin/env python3
"""
Email Clarity Rewriter Agent (Streamlit + DeepSeek)

Requirements (install with pip):
- streamlit
- python-dotenv
- openai  (DeepSeek uses an OpenAI-compatible API)

Example .env (at repository root):
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1   # optional override
DEEPSEEK_MODEL=deepseek-chat                   # optional override
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_environment() -> None:
    """Load environment variables from the root .env file."""
    if not DOTENV_PATH.is_file():
        raise FileNotFoundError(f".env not found at {DOTENV_PATH}")
    load_dotenv(DOTENV_PATH)


def _get_deepseek_key() -> str:
    """Return DeepSeek API key from environment."""
    load_environment()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("DEEPSEEK_API_KEY not set in .env")
    return key


def _get_deepseek_client() -> OpenAI:
    """Return an OpenAI-compatible client configured for DeepSeek."""
    api_key = _get_deepseek_key()
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def _get_model_name() -> str:
    """Return DeepSeek model name (override with DEEPSEEK_MODEL)."""
    load_environment()
    return os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


def build_prompt(
    email_text: str,
    tone: str,
    shorten: bool,
    highlight_unclear: bool,
    improve_structure: bool,
) -> str:
    """Build user prompt for the LLM."""
    flags: list[str] = []
    if shorten:
        flags.append("shorten_email")
    if highlight_unclear:
        flags.append("highlight_unclear_sentences")
    if improve_structure:
        flags.append("improve_structure")

    flags_str = ", ".join(flags) if flags else "none"

    return (
        f"Tone: {tone}\n"
        f"Options: {flags_str}\n\n"
        "Original email:\n"
        f"{email_text}\n\n"
        "Return output strictly in this format:\n\n"
        "REWRITTEN_EMAIL:\n"
        "<rewritten email text>\n"
        "IMPROVEMENTS_MADE:\n\n"
        "Bullet points explaining changes\n\n"
        "CLARITY_ISSUES:\n\n"
        "Bullet points identifying unclear sentences (or None detected)\n"
    )


def call_deepseek_api(prompt: str, timeout_sec: int = 60) -> str:
    """Call DeepSeek chat completion API and return raw text response."""
    client = _get_deepseek_client()
    model = _get_model_name()
    system_message = (
        "You are a professional communication assistant specialized in business and "
        "academic email refinement.\n\n"
        "Your tasks:\n\n"
        "Rewrite the email according to the selected tone.\n"
        "Improve clarity, structure, and conciseness.\n"
        "Remove redundancy and informal phrasing.\n"
        "Maintain the original intent.\n"
        "If requested, shorten the email without losing meaning.\n"
        "If requested, identify unclear or ambiguous sentences and explain why they are unclear.\n"
        "Provide output in structured format with clear section headers.\n"
        "Never change factual meaning. Never invent details.\n"
        "Return output in the exact format specified by the user."
    )

    try:
        logger.info("Calling DeepSeek model=%s", model)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=800,
            timeout=timeout_sec,
        )
        content = (completion.choices[0].message.content or "").strip()
        logger.info("DeepSeek call succeeded (%s chars)", len(content))
        return content
    except Exception as exc:
        logger.error("DeepSeek API call failed: %s", exc)
        raise RuntimeError("Error calling DeepSeek API") from exc


def parse_llm_response(raw: str) -> Dict[str, str]:
    """
    Parse LLM response into sections:
    - rewritten_email
    - improvements_made
    - clarity_issues
    """
    text = raw.strip()

    def extract_section(name: str, source: str) -> str:
        pattern = rf"{name}:\s*(.*?)(?=(REWRITTEN_EMAIL:|IMPROVEMENTS_MADE:|CLARITY_ISSUES:|$))"
        match = re.search(pattern, source, flags=re.DOTALL | re.IGNORECASE)
        if not match:
            return ""
        return match.group(1).strip()

    rewritten = extract_section("REWRITTEN_EMAIL", text)
    improvements = extract_section("IMPROVEMENTS_MADE", text)
    clarity = extract_section("CLARITY_ISSUES", text)

    if not rewritten:
        raise ValueError("Could not parse rewritten email from model response.")

    return {
        "rewritten_email": rewritten,
        "improvements_made": improvements or "No explicit improvements listed.",
        "clarity_issues": clarity or "None detected",
    }


def _split_sentences(text: str) -> list[str]:
    """Very simple sentence splitter."""
    fragments = re.split(r"[.!?]+", text)
    return [s.strip() for s in fragments if s.strip()]


def _tokenize_words(text: str) -> list[str]:
    """Basic word tokenizer."""
    return re.findall(r"\b\w+\b", text)


def compute_readability_score(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    Compute a lightweight readability estimate based on average sentence length.

    Returns:
        label: "Easy", "Moderate", or "Complex"
        details: metrics used in the estimation
    """
    sentences = _split_sentences(text)
    words = _tokenize_words(text)

    num_sentences = max(len(sentences), 1)
    num_words = len(words)
    avg_sentence_len = num_words / num_sentences if num_sentences else 0.0

    if avg_sentence_len <= 12:
        label = "Easy"
    elif avg_sentence_len <= 20:
        label = "Moderate"
    else:
        label = "Complex"

    details: Dict[str, Any] = {
        "num_sentences": num_sentences,
        "num_words": num_words,
        "avg_sentence_length": round(avg_sentence_len, 2),
    }
    return label, details


def render_streamlit_ui() -> None:
    """Render the Streamlit interface."""
    st.set_page_config(page_title="Email Clarity Rewriter Agent", page_icon="", layout="centered")
    st.title("Email Clarity Rewriter Agent")
    st.caption("Rewrite and clarify emails using DeepSeek with tone and structure control.")

    st.markdown("---")

    col_tone, col_options = st.columns([1, 1])

    with col_tone:
        tone = st.selectbox(
            "Tone",
            options=["Professional", "Polite", "Assertive", "Concise"],
            index=0,
        )

    with col_options:
        shorten = st.checkbox("Shorten the email")
        highlight_unclear = st.checkbox("Highlight unclear sentences")
        improve_structure = st.checkbox("Improve structure", value=True)

    email_text = st.text_area(
        "Original email",
        height=260,
        placeholder="Paste the email you want to improve...",
    )

    st.markdown("---")

    run_button = st.button("Rewrite Email", type="primary", use_container_width=True)

    if run_button:
        if not email_text.strip():
            st.error("Please paste an email before rewriting.")
            return

        with st.spinner("Rewriting email with DeepSeek..."):
            try:
                prompt = build_prompt(
                    email_text=email_text.strip(),
                    tone=tone,
                    shorten=shorten,
                    highlight_unclear=highlight_unclear,
                    improve_structure=improve_structure,
                )
                raw_response = call_deepseek_api(prompt)
                parsed = parse_llm_response(raw_response)
            except (ValueError, RuntimeError) as exc:
                st.error(str(exc))
                return

        rewritten = parsed["rewritten_email"]
        improvements = parsed["improvements_made"]
        clarity = parsed["clarity_issues"]

        st.subheader("A) Rewritten Email")
        st.text_area(
            "Rewritten email",
            value=rewritten,
            height=260,
            label_visibility="collapsed",
        )

        readability_label, details = compute_readability_score(rewritten)

        with st.expander("B) Summary of Improvements Made", expanded=True):
            st.write(improvements)

        if highlight_unclear:
            with st.expander("C) Clarity Issues Found", expanded=True):
                st.write(clarity)

        st.subheader("D) Readability Score")
        col_r1, col_r2, col_r3 = st.columns(3)
        col_r1.metric("Estimated readability", readability_label)
        col_r2.metric("Sentences", str(details["num_sentences"]))
        col_r3.metric("Words", str(details["num_words"]))
        st.caption(f"Average sentence length: {details['avg_sentence_length']} words")

        st.markdown("---")

        col_dl, _ = st.columns([1, 1])
        with col_dl:
            st.download_button(
                "Download rewritten email as .txt",
                data=rewritten,
                file_name="rewritten_email.txt",
                mime="text/plain",
                use_container_width=True,
            )


def main() -> None:
    """Entry point for the Streamlit app."""
    render_streamlit_ui()


if __name__ == "__main__":
    main()

