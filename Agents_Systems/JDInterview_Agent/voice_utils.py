"""
Voice helpers for interview practice: speech-to-text (Whisper) and text-to-speech (gTTS).
Uses OPENAI_API_KEY from env (repo root .env) for transcription.
"""

import io
import os

# Ensure repo .env is loaded before reading OPENAI_API_KEY
from config import settings  # noqa: F401

from openai import OpenAI
from gtts import gTTS


def get_openai_client() -> OpenAI | None:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        return None
    return OpenAI(api_key=key)


def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    """Transcribe audio bytes to text using OpenAI Whisper. Requires OPENAI_API_KEY."""
    client = get_openai_client()
    if not client:
        raise RuntimeError("OPENAI_API_KEY not set. Add it to the repo root .env for voice input.")
    f = io.BytesIO(audio_bytes)
    f.name = filename
    result = client.audio.transcriptions.create(model="whisper-1", file=f)
    return (result.text or "").strip()


def text_to_speech_bytes(text: str) -> bytes:
    """Convert text to speech and return MP3 bytes (gTTS). No API key required."""
    tts = gTTS(text=text)
    buf = io.BytesIO()
    tts.write_to_fp(buf)
    buf.seek(0)
    return buf.read()
