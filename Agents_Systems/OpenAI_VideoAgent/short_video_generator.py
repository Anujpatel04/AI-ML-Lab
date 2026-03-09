#!/usr/bin/env python3
"""
OpenAI Sora short video generator (20–30 seconds).
Uses OPENAI_API_KEY from project .env. Generates 2–3 clips and concatenates with ffmpeg.
Uses REST API directly so it works with any openai package version.
"""

import json
import os
import sys
import time
import subprocess
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

BASE_URL = "https://api.openai.com/v1"

# Load .env from project root (parent of AI_AGENTS)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"

def load_env():
    """Load OPENAI_API_KEY from .env without adding project to path."""
    if not DOTENV_PATH.is_file():
        raise FileNotFoundError(f".env not found at {DOTENV_PATH}")
    env = {}
    with open(DOTENV_PATH) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip()
            if v.startswith('"') and v.endswith('"'):
                v = v[1:-1].replace('\\"', '"')
            elif v.startswith("'") and v.endswith("'"):
                v = v[1:-1].replace("\\'", "'")
            env[k] = v
    return env

def get_api_key():
    env = load_env()
    key = env.get("OPENAI_API_KEY")
    if not key:
        raise ValueError("OPENAI_API_KEY not set in .env")
    return key

def _api_call(req: urllib.request.Request) -> bytes:
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        body = e.read().decode() if e.fp else ""
        try:
            err = json.loads(body)
            msg = err.get("error", {}).get("message", body) if isinstance(err.get("error"), dict) else body
        except Exception:
            msg = body or str(e)
        raise RuntimeError(f"API error {e.code}: {msg}") from e

def create_video(api_key: str, prompt: str, seconds: str = "12", model: str = "sora-2", size: str = "1280x720") -> dict:
    """Start a video generation job via POST /v1/videos (multipart/form-data)."""
    boundary = "----WebKitFormBoundary" + os.urandom(16).hex()
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="prompt"\r\n\r\n{prompt}\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="model"\r\n\r\n{model}\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="seconds"\r\n\r\n{seconds}\r\n'
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="size"\r\n\r\n{size}\r\n'
        f"--{boundary}--\r\n"
    ).encode("utf-8")
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": f"multipart/form-data; boundary={boundary}"}
    req = urllib.request.Request(f"{BASE_URL}/videos", data=body, headers=headers, method="POST")
    return json.loads(_api_call(req).decode())

def get_video_status(api_key: str, video_id: str) -> dict:
    """GET /v1/videos/{video_id}"""
    req = urllib.request.Request(
        f"{BASE_URL}/videos/{video_id}",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    return json.loads(_api_call(req).decode())

def wait_for_video(api_key: str, video_id: str, poll_interval: int = 15) -> dict:
    """Poll until job is completed or failed."""
    while True:
        video = get_video_status(api_key, video_id)
        status = video.get("status", "")
        if status == "completed":
            return video
        if status == "failed":
            err = video.get("error") or {}
            msg = err.get("message", str(err))
            raise RuntimeError(f"Video generation failed: {msg}")
        progress = video.get("progress", 0)
        print(f"  Status: {status}, progress: {progress}%")
        time.sleep(poll_interval)

def download_video(api_key: str, video_id: str) -> bytes:
    """Download MP4: GET /v1/videos/{video_id}/content"""
    req = urllib.request.Request(
        f"{BASE_URL}/videos/{video_id}/content",
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    return _api_call(req)

def concatenate_mp4s(paths: list, out_path: str) -> bool:
    """Concatenate MP4 files using ffmpeg. Returns True on success."""
    if not paths:
        return False
    if len(paths) == 1:
        import shutil
        shutil.copy(paths[0], out_path)
        return True
    list_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    try:
        for p in paths:
            list_file.write(f"file '{os.path.abspath(p)}'\n")
        list_file.close()
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_file.name, "-c", "copy", out_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False
    finally:
        try:
            os.unlink(list_file.name)
        except Exception:
            pass

def generate_short_video(
    prompt: str,
    output_path: str = "output.mp4",
    duration_seconds: int = 24,
    model: str = "sora-2",
    size: str = "1280x720",
    poll_interval: int = 15,
    progress_callback=None,
):
    """
    Generate a short video (15–30 seconds) from a text prompt.

    duration_seconds: target total length (15–30). Uses 4s, 8s, 12s clips.
    progress_callback: optional callable(message: str) for UI updates (e.g. Streamlit).
    """
    if duration_seconds < 15:
        duration_seconds = 15
    if duration_seconds > 30:
        duration_seconds = 30

    def report(msg: str):
        if progress_callback:
            progress_callback(msg)
        else:
            print(msg)

    # Clip lengths: use 12s, 8s, 4s to reach target
    clips = []
    remaining = duration_seconds
    while remaining >= 12:
        clips.append("12")
        remaining -= 12
    if remaining >= 8:
        clips.append("8")
        remaining -= 8
    elif remaining >= 4:
        clips.append("4")
        remaining -= 4
    if not clips:
        clips = ["12"]

    api_key = get_api_key()
    total_clips = len(clips)
    report(f"Generating {total_clips} clip(s): {', '.join(clips)}s → ~{sum(int(c) for c in clips)}s total")
    report(f"Model: {model}, size: {size}")

    part_prompts = []
    if total_clips == 1:
        part_prompts = [prompt]
    else:
        for i in range(total_clips):
            part_prompts.append(
                f"[Part {i+1}/{total_clips}] {prompt}" if total_clips > 1 else prompt
            )

    temp_dir = tempfile.mkdtemp()
    clip_paths = []
    try:
        for i, (sec, part_prompt) in enumerate(zip(clips, part_prompts)):
            report(f"Clip {i+1}/{total_clips} ({sec}s)...")
            job = create_video(api_key, part_prompt, seconds=sec, model=model, size=size)
            video_id = job["id"]
            wait_for_video(api_key, video_id, poll_interval=poll_interval)
            data = download_video(api_key, video_id)
            path = os.path.join(temp_dir, f"clip_{i}.mp4")
            with open(path, "wb") as f:
                f.write(data)
            clip_paths.append(path)
            report(f"Saved clip {i+1}.")

        out_abs = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
        if concatenate_mp4s(clip_paths, out_abs):
            report(f"Done. Video saved to: {out_abs}")
        else:
            with open(clip_paths[0], "rb") as f:
                with open(out_abs, "wb") as out:
                    out.write(f.read())
            report("ffmpeg not found; saved single clip.")
    finally:
        for p in clip_paths:
            try:
                os.unlink(p)
            except Exception:
                pass
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass
    return out_abs

def main():
    if len(sys.argv) < 2:
        print("Usage: python short_video_generator.py <prompt> [output.mp4]")
        print("Example: python short_video_generator.py 'A calico cat playing piano on stage, soft lighting' my_video.mp4")
        sys.exit(1)
    prompt = sys.argv[1]
    output = sys.argv[2] if len(sys.argv) > 2 else "output.mp4"
    generate_short_video(prompt, output_path=output)


# ----- Streamlit frontend -----
def run_streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="Short Video Generator", page_icon="🎬", layout="centered")
    st.title("🎬 OpenAI Short Video Generator")
    st.caption("Generate 15–30 second videos with Sora via your prompt")

    prompt = st.text_area(
        "Video prompt",
        placeholder="e.g. A calico cat playing piano on stage, soft lighting, cinematic",
        height=100,
        help="Describe the scene, action, and style for your video.",
    )
    duration = st.slider(
        "Duration (seconds)",
        min_value=15,
        max_value=30,
        value=24,
        step=1,
        help="Choose between 15 and 30 seconds. Longer videos take more time.",
    )
    col1, col2 = st.columns(2)
    with col1:
        model = st.selectbox("Model", options=["sora-2", "sora-2-pro"], index=0, help="sora-2 is faster, sora-2-pro is higher quality.")
    with col2:
        size = st.selectbox(
            "Resolution",
            options=["1280x720", "720x1280", "1024x1792", "1792x1024"],
            index=0,
        )

    if st.button("Generate video", type="primary", use_container_width=True):
        if not prompt or not prompt.strip():
            st.error("Please enter a video prompt.")
            return
        progress_placeholder = st.empty()
        log_placeholder = st.container()
        messages = []

        def on_progress(msg: str):
            messages.append(msg)
            with log_placeholder:
                for m in messages:
                    st.text(m)

        with progress_placeholder:
            with st.spinner("Generating your video… This may take several minutes."):
                try:
                    out_path = generate_short_video(
                        prompt.strip(),
                        output_path=os.path.join(tempfile.gettempdir(), f"streamlit_video_{int(time.time())}.mp4"),
                        duration_seconds=duration,
                        model=model,
                        size=size,
                        progress_callback=on_progress,
                    )
                except Exception as e:
                    st.error(str(e))
                    return

        progress_placeholder.empty()
        st.success("Video ready!")
        with open(out_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
        st.download_button(
            "Download MP4",
            data=video_bytes,
            file_name="generated_video.mp4",
            mime="video/mp4",
            use_container_width=True,
        )


if __name__ == "__main__":
    # CLI: python short_video_generator.py "prompt" [output.mp4]
    # Web UI: streamlit run short_video_generator.py
    if len(sys.argv) >= 2 and not sys.argv[1].startswith("-"):
        main()
    else:
        run_streamlit_app()
