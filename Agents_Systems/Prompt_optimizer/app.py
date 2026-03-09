import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


ROOT_ENV_PATH = "/Users/anuj/Desktop/Anuj-AI-ML-Lab/.env"
DEFAULT_MODEL = "gpt-4o-mini"


SYSTEM_PROMPT = """You are a Prompt Optimization Agent.
Given a weak prompt, return a significantly improved, high-quality prompt.
Output must be valid JSON that strictly follows this schema:
{
  "original_prompt": "<string>",
  "detected_intent": "<string>",
  "issues_found": ["<issue1>", "<issue2>"],
  "optimized_prompt": "<string>",
  "optimization_notes": "<short explanation>"
}
Return only JSON. Do not add extra keys or commentary.
"""

ANSWER_SYSTEM_PROMPT = "You are a helpful assistant. Follow the user's prompt precisely."


REQUIRED_FIELDS = {
    "original_prompt": str,
    "detected_intent": str,
    "issues_found": list,
    "optimized_prompt": str,
    "optimization_notes": str,
}


def load_env() -> None:
    load_dotenv(ROOT_ENV_PATH, override=True)


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=api_key)


def get_model_name() -> str:
    return os.getenv("OPENAI_MODEL", DEFAULT_MODEL)


def validate_output(data: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Output must be a JSON object.")

    missing = [key for key in REQUIRED_FIELDS if key not in data]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    for key, expected_type in REQUIRED_FIELDS.items():
        value = data.get(key)
        if not isinstance(value, expected_type):
            raise ValueError(f"Field '{key}' must be {expected_type.__name__}.")

    if not all(isinstance(item, str) for item in data["issues_found"]):
        raise ValueError("All items in 'issues_found' must be strings.")

    return data


def extract_json(text: str) -> Dict[str, Any]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in response.")
    return json.loads(match.group(0))


def optimize_prompt(raw_prompt: str) -> Dict[str, Any]:
    if not raw_prompt or not raw_prompt.strip():
        raise ValueError("Input prompt must be a non-empty string.")

    client = get_openai_client()
    model = get_model_name()

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": raw_prompt.strip()},
        ],
    )

    content = response.choices[0].message.content or ""
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = extract_json(content)

    return validate_output(data)


def generate_answer(prompt: str) -> str:
    if not prompt or not prompt.strip():
        raise ValueError("Prompt must be a non-empty string.")

    client = get_openai_client()
    model = get_model_name()

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": prompt.strip()},
        ],
    )

    return (response.choices[0].message.content or "").strip()


def run_streamlit() -> None:
    st.set_page_config(page_title="Prompt Optimization Agent", layout="centered")
    st.title("Prompt Optimization Agent")
    st.write(
        "Paste a prompt to get a clearer, structured, and optimized version."
    )

    load_env()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY is not set in the environment.")
        st.info(f"Expected .env location: {ROOT_ENV_PATH}")
        st.stop()

    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    with st.form("prompt_form"):
        raw_prompt = st.text_area(
            "Raw prompt",
            placeholder="Enter the prompt you want to optimize...",
            height=160,
        )
        submitted = st.form_submit_button("Optimize")

    if submitted:
        try:
            st.session_state.last_result = optimize_prompt(raw_prompt)
        except Exception as exc:
            st.session_state.last_result = {"error": str(exc)}

    if st.session_state.last_result:
        if "error" in st.session_state.last_result:
            st.error(f"Error: {st.session_state.last_result['error']}")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Prompt")
                st.write(st.session_state.last_result["original_prompt"])
            with col2:
                st.subheader("Optimized Prompt")
                st.write(st.session_state.last_result["optimized_prompt"])

            with st.spinner("Generating answers for both prompts..."):
                original_answer = generate_answer(
                    st.session_state.last_result["original_prompt"]
                )
                optimized_answer = generate_answer(
                    st.session_state.last_result["optimized_prompt"]
                )

            ans_col1, ans_col2 = st.columns(2)
            with ans_col1:
                st.subheader("Answer to Original Prompt")
                st.write(original_answer)
            with ans_col2:
                st.subheader("Answer to Optimized Prompt")
                st.write(optimized_answer)

            st.subheader("Full Output (JSON)")
            st.json(st.session_state.last_result)


def run_cli() -> None:
    parser = argparse.ArgumentParser(description="Prompt Optimization Agent")
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        help="Raw prompt string to optimize.",
    )
    args = parser.parse_args()

    raw_prompt = args.prompt
    if not raw_prompt:
        if not sys.stdin.isatty():
            raw_prompt = sys.stdin.read().strip()
        else:
            raw_prompt = input("Enter a prompt to optimize: ").strip()

    load_env()
    result = optimize_prompt(raw_prompt)
    print(json.dumps(result, indent=2))


def is_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime import exists

        return exists()
    except Exception:
        return bool(getattr(st, "_is_running_with_streamlit", False))


if __name__ == "__main__":
    if is_streamlit_runtime():
        run_streamlit()
    else:
        run_cli()
