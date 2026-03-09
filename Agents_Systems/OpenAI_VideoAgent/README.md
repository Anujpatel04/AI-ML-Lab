# OpenAI Video Agent

>Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab/tree/main).

Short video generator (15–30 seconds) using the OpenAI Sora API. Supports a Streamlit web UI and CLI. Video is built from one or more Sora clips (4s, 8s, or 12s each) and concatenated when duration exceeds a single clip.

## Requirements

- Python 3.9+
- `OPENAI_API_KEY` in the project root `.env` (parent of `AI_AGENTS`)
- Optional: `ffmpeg` for concatenating multiple clips into one file (single-clip output works without it)
- Sora API access (invite-only; request in the OpenAI dashboard if needed)

## Installation

```bash
cd AI_AGENTS/OpenAI_VideoAgent
pip install -r requirements.txt
```

## Usage

**Web UI (Streamlit)**

```bash
streamlit run short_video_generator.py
```

Use the form to enter a prompt, set duration (15–30 seconds), choose model and resolution, then generate. Progress is shown in the app; the result can be played and downloaded.

**CLI**

```bash
python short_video_generator.py "Your video prompt here" [output.mp4]
```

Examples:

```bash
python short_video_generator.py "A calico cat playing piano on stage, soft lighting"
python short_video_generator.py "Wide shot of a red kite in a park, golden hour" my_video.mp4
```

## Configuration

- **Models:** `sora-2` (faster), `sora-2-pro` (higher quality)
- **Resolutions:** 1280x720, 720x1280, 1024x1792, 1792x1024 (UI only; CLI uses 1280x720)
- **Duration:** 15–30 seconds; the script picks 4s/8s/12s clips to match

