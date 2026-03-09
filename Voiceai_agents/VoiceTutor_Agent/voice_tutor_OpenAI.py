import io
import json
import os
from datetime import datetime

import streamlit as st
from gtts import gTTS
from openai import OpenAI


def load_env(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            raw = line.strip()
            if not raw or raw.startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


def find_env(start_dir: str) -> str | None:
    current = os.path.abspath(start_dir)
    while True:
        candidate = os.path.join(current, ".env")
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


def get_client() -> OpenAI:
    env_path = find_env(os.path.dirname(__file__))
    if env_path:
        load_env(env_path)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment or .env")
    return OpenAI(api_key=api_key)


def transcribe_audio(client: OpenAI, audio_bytes: bytes, filename: str) -> str:
    audio_file = io.BytesIO(audio_bytes)
    audio_file.name = filename
    transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
    )
    return transcript.text.strip()


def synthesize_speech(text: str) -> bytes:
    tts = gTTS(text=text)
    audio_fp = io.BytesIO()
    tts.write_to_fp(audio_fp)
    audio_fp.seek(0)
    return audio_fp.read()


def generate_explanation_and_followup(
    client: OpenAI, user_prompt: str, difficulty: str
) -> tuple[str, str]:
    system_prompt = (
        "You are a concise voice tutor. "
        "Generate an explanation at the requested difficulty level and ask exactly one follow-up question. "
        "Output ONLY valid JSON with keys: explanation, followup_question."
    )
    user_msg = (
        f"Difficulty: {difficulty}\n"
        f"User request: {user_prompt}\n"
        "Difficulty guidelines:\n"
        "- Beginner: simple language, analogies\n"
        "- Intermediate: intuition + light math\n"
        "- Advanced: formulas, gradients, chain rule\n"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    )
    content = response.choices[0].message.content.strip()
    data = json.loads(content)
    explanation = data["explanation"].strip()
    followup = data["followup_question"].strip().rstrip("?") + "?"
    return explanation, followup


def evaluate_answer(
    client: OpenAI, followup_question: str, user_answer: str
) -> str:
    system_prompt = (
        "Evaluate if the user's answer shows understanding. "
        "Respond with exactly one word: weak or good."
    )
    user_msg = (
        f"Question: {followup_question}\n"
        f"Answer: {user_answer}\n"
        "Return only: weak or good."
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    )
    verdict = response.choices[0].message.content.strip().lower()
    return "good" if "good" in verdict else "weak"


def adjust_difficulty(current: str, verdict: str) -> str:
    order = ["Beginner", "Intermediate", "Advanced"]
    idx = order.index(current)
    if verdict == "good" and idx < len(order) - 1:
        return order[idx + 1]
    if verdict == "weak" and idx > 0:
        return order[idx - 1]
    return current


def generate_adjusted_explanation(
    client: OpenAI, user_prompt: str, difficulty: str
) -> str:
    system_prompt = (
        "You are a concise voice tutor. "
        "Give a short adjusted explanation only. No questions."
    )
    user_msg = (
        f"Difficulty: {difficulty}\n"
        f"User request: {user_prompt}\n"
        "Difficulty guidelines:\n"
        "- Beginner: simple language, analogies\n"
        "- Intermediate: intuition + light math\n"
        "- Advanced: formulas, gradients, chain rule\n"
    )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ],
    )
    return response.choices[0].message.content.strip()


st.set_page_config(page_title="Voice Tutor", page_icon="🎧")
st.title("Voice Tutor / Study Buddy")

if "difficulty" not in st.session_state:
    st.session_state.difficulty = "Beginner"
if "awaiting_followup" not in st.session_state:
    st.session_state.awaiting_followup = False
if "last_followup" not in st.session_state:
    st.session_state.last_followup = ""
if "last_prompt" not in st.session_state:
    st.session_state.last_prompt = ""
if "last_transcript" not in st.session_state:
    st.session_state.last_transcript = ""
if "last_response" not in st.session_state:
    st.session_state.last_response = ""
if "last_audio" not in st.session_state:
    st.session_state.last_audio = b""

difficulty_options = ["Beginner", "Intermediate", "Advanced"]
selected_difficulty = st.selectbox(
    "Select explanation level",
    difficulty_options,
    index=difficulty_options.index(st.session_state.difficulty),
    disabled=st.session_state.awaiting_followup,
)
if not st.session_state.awaiting_followup:
    st.session_state.difficulty = selected_difficulty

st.caption(f"Current difficulty: {st.session_state.difficulty}")

audio_input = st.audio_input("Record your question or answer")
uploaded_audio = st.file_uploader(
    "Or upload audio", type=["wav", "mp3", "m4a", "ogg", "webm"]
)

if st.button("Process audio"):
    selected_audio = audio_input or uploaded_audio
    if not selected_audio:
        st.warning("Please record or upload an audio file.")
    else:
        audio_bytes = (
            selected_audio.getvalue()
            if hasattr(selected_audio, "getvalue")
            else selected_audio
        )
        filename = getattr(selected_audio, "name", f"audio_{datetime.utcnow().timestamp()}.wav")
        try:
            client = get_client()
            transcript = transcribe_audio(client, audio_bytes, filename)
            st.session_state.last_transcript = transcript

            if not st.session_state.awaiting_followup:
                st.session_state.last_prompt = transcript
                explanation, followup = generate_explanation_and_followup(
                    client, transcript, st.session_state.difficulty
                )
                response_text = f"{explanation}\n\nFollow-up question: {followup}"
                st.session_state.last_followup = followup
                st.session_state.awaiting_followup = True
            else:
                verdict = evaluate_answer(
                    client, st.session_state.last_followup, transcript
                )
                st.session_state.difficulty = adjust_difficulty(
                    st.session_state.difficulty, verdict
                )
                adjusted = generate_adjusted_explanation(
                    client, st.session_state.last_prompt, st.session_state.difficulty
                )
                response_text = (
                    f"Based on your answer, here's a {st.session_state.difficulty.lower()} explanation:\n\n"
                    f"{adjusted}"
                )
                st.session_state.awaiting_followup = False

            audio_response = synthesize_speech(response_text)
            st.session_state.last_response = response_text
            st.session_state.last_audio = audio_response
        except Exception as exc:
            st.error(str(exc))

if st.session_state.last_transcript:
    st.subheader("Transcribed Text")
    st.write(st.session_state.last_transcript)

if st.session_state.last_response:
    st.subheader("Tutor Response")
    st.write(st.session_state.last_response)
    st.audio(st.session_state.last_audio, format="audio/mp3")
