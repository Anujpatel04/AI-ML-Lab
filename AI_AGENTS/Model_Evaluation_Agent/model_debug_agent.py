#!/usr/bin/env python3
"""
Model Debug Agent: overfitting/underfitting detection, convergence analysis,
and hyperparameter/architecture recommendations via deterministic analysis and OpenAI.
"""

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"


class ModelDebugInput(BaseModel):
    architecture: str = Field(..., description="Model architecture description")
    training_logs: str = Field(..., description="Training logs text")
    train_loss: list[float] = Field(..., description="Training loss per epoch")
    val_loss: list[float] = Field(..., description="Validation loss per epoch")


def load_env() -> None:
    """Load environment from repo root .env."""
    if not DOTENV_PATH.is_file():
        raise FileNotFoundError(f".env not found at {DOTENV_PATH}")
    load_dotenv(DOTENV_PATH)


def get_deepseek_api_key() -> str:
    load_env()
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("DEEPSEEK_API_KEY not set in .env")
    return key


def get_llm_client() -> OpenAI:
    """
    Create an OpenAI-compatible client pointed at DeepSeek.

    DeepSeek provides an OpenAI-compatible API; set `DEEPSEEK_BASE_URL` to override.
    """
    load_env()
    api_key = get_deepseek_api_key()
    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def get_llm_model() -> str:
    """DeepSeek model name (override with DEEPSEEK_MODEL)."""
    load_env()
    return os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")


def _validate_arrays(train_loss: list[float], val_loss: list[float]) -> None:
    if len(train_loss) != len(val_loss):
        raise ValueError("train_loss and val_loss must have the same length")
    if not train_loss:
        raise ValueError("train_loss and val_loss must be non-empty")
    for i, (t, v) in enumerate(zip(train_loss, val_loss)):
        if not isinstance(t, (int, float)) or not isinstance(v, (int, float)):
            raise ValueError(f"Loss values must be numeric at index {i}")


def analyze_losses(
    train_loss: list[float],
    val_loss: list[float],
) -> dict[str, Any]:
    """
    Deterministic analysis: overfitting, underfitting, instability,
    generalization gap, best epoch, convergence speed. No LLM.
    """
    _validate_arrays(train_loss, val_loss)
    n = len(train_loss)
    train_final = float(train_loss[-1])
    val_final = float(val_loss[-1])
    gap = val_final - train_final
    best_epoch = int(val_loss.index(min(val_loss))) + 1

    overfitting = False
    underfitting = False
    instability = False

    if n >= 3:
        train_decreasing = train_loss[-1] < train_loss[0]
        val_increasing = val_loss[-1] > val_loss[0]
        if train_decreasing and val_increasing:
            overfitting = True
        if gap > 0.5 * max(train_final, val_final, 1e-6):
            overfitting = True

    both_high = train_final > 0.5 and val_final > 0.5
    flat_train = n >= 3 and max(train_loss) - min(train_loss) < 0.05
    flat_val = n >= 3 and max(val_loss) - min(val_loss) < 0.05
    if both_high and (flat_train or flat_val):
        underfitting = True

    if n >= 4:
        val_changes = [val_loss[i] - val_loss[i - 1] for i in range(1, n)]
        sign_changes = sum(1 for i in range(1, len(val_changes)) if val_changes[i] * val_changes[i - 1] < 0)
        if sign_changes >= n // 2:
            instability = True
        if any(v > 10 for v in val_loss) or any(t > 10 for t in train_loss):
            instability = True

    improvement = train_loss[0] - train_loss[-1] if train_loss[0] > train_loss[-1] else 0.0
    convergence_speed = improvement / n if n else 0.0

    return {
        "overfitting": overfitting,
        "underfitting": underfitting,
        "instability": instability,
        "generalization_gap": round(gap, 6),
        "best_epoch": best_epoch,
        "final_train_loss": round(train_final, 6),
        "final_val_loss": round(val_final, 6),
        "convergence_speed_est": round(convergence_speed, 6),
        "num_epochs": n,
    }


def build_prompt(
    raw_input: ModelDebugInput,
    analysis: dict[str, Any],
) -> str:
    """Build user message for LLM with raw input and deterministic analysis."""
    analysis_str = json.dumps(analysis, indent=2)
    return f"""## Model architecture
{raw_input.architecture}

## Training logs
{raw_input.training_logs}

## Deterministic analysis (pre-computed)
{analysis_str}

## Loss arrays (last 5 epochs shown)
train_loss (last 5): {raw_input.train_loss[-5:]}
val_loss (last 5): {raw_input.val_loss[-5:]}

Respond with a single JSON object (no markdown, no code fence) containing exactly: training_diagnosis (overfitting, underfitting, instability, convergence_quality, generalization_gap), hyperparameter_recommendations (list of {{parameter, suggestion, recommended_range}}), architecture_improvements (list of strings), regularization_strategies (list of strings), optimization_strategies (list of strings), data_recommendations (list of strings), priority_action_plan (ordered list of steps). Use the pre-computed analysis and be specific (learning rate ranges, dropout %, weight decay, batch size). convergence_quality must be one of: poor, moderate, good."""


def get_system_prompt() -> str:
    return """You are a Senior ML Research Engineer. Your task is to diagnose training and return a single JSON object.

Rules:
- Be mathematically grounded; avoid generic advice.
- Suggest specific hyperparameters: learning rate ranges (e.g. 1e-4 to 1e-3), dropout percentages, weight decay ranges, batch size adjustments.
- Suggest concrete architecture changes (layers, width, depth, normalization).
- Suggest regularization (L2, dropout, early stopping, data augmentation).
- Suggest optimizer alternatives (AdamW, SGD with momentum, learning rate schedules).
- Prioritize recommendations in priority_action_plan (Step 1, Step 2, ...).
- Return only valid JSON; no markdown code block, no explanation outside JSON.
- training_diagnosis must include: overfitting (bool), underfitting (bool), instability (bool), convergence_quality ("poor"|"moderate"|"good"), generalization_gap (float).
- hyperparameter_recommendations: list of { "parameter": str, "suggestion": str, "recommended_range": str }."""


def call_openai(
    user_content: str,
    max_retries: int = 2,
    timeout_sec: int = 120,
) -> str:
    """Call DeepSeek (OpenAI-compatible) with retry and timeout. Returns response content."""
    client = get_llm_client()
    model = get_llm_model()
    system = get_system_prompt()
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            logger.info("Calling LLM (attempt %s)", attempt + 1)
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
                timeout=timeout_sec,
            )
            content = (resp.choices[0].message.content or "").strip()
            logger.info("Received response (%s chars)", len(content))
            return content
        except Exception as e:
            last_error = e
            logger.warning("LLM call failed: %s", e)
            if attempt == max_retries:
                raise last_error from e
    raise last_error or RuntimeError("LLM call failed")


def validate_json(raw: str) -> dict[str, Any]:
    """Extract JSON from response and validate structure. Retries not applied here (caller retries)."""
    cleaned = raw.strip()
    match = re.search(r"\{[\s\S]*\}", cleaned)
    if match:
        cleaned = match.group(0)
    try:
        out = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e
    if not isinstance(out, dict):
        raise ValueError("Output must be a JSON object")
    for key in (
        "training_diagnosis",
        "hyperparameter_recommendations",
        "architecture_improvements",
        "regularization_strategies",
        "optimization_strategies",
        "data_recommendations",
        "priority_action_plan",
    ):
        if key not in out:
            raise ValueError(f"Missing required key: {key}")
    return out


def run_debug_agent(input_data: ModelDebugInput) -> dict[str, Any]:
    """
    Run deterministic analysis, call LLM, validate JSON. Returns structured dict.
    """
    logger.info("Running deterministic analysis")
    analysis = analyze_losses(input_data.train_loss, input_data.val_loss)
    logger.info("Analysis: overfitting=%s, underfitting=%s, instability=%s", analysis["overfitting"], analysis["underfitting"], analysis["instability"])

    user_content = build_prompt(input_data, analysis)

    for attempt in range(3):
        try:
            content = call_openai(user_content)
            result = validate_json(content)
            result["_deterministic_analysis"] = analysis
            logger.info("Successfully produced structured output")
            return result
        except ValueError as e:
            logger.warning("JSON validation failed (attempt %s): %s", attempt + 1, e)
            if attempt == 2:
                raise
            user_content += "\n\nPrevious response was invalid JSON. Return only a single valid JSON object, no markdown."


def plot_loss_curves(
    train_loss: list[float],
    val_loss: list[float],
    out_path: str | Path = "loss_curves.png",
) -> Path:
    """Save matplotlib loss curve PNG. Returns path."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(out_path)
    plt.figure(figsize=(8, 5))
    epochs = range(1, len(train_loss) + 1)
    plt.plot(epochs, train_loss, label="Train loss", color="C0")
    plt.plot(epochs, val_loss, label="Val loss", color="C1")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training and validation loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path.resolve()


def _parse_floats(s: str) -> list[float]:
    return [float(x.strip()) for x in s.replace(",", " ").split() if x.strip()]


def run_streamlit_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="Model Debug Agent", page_icon="", layout="centered")
    st.title("Model Debug Agent")
    st.caption("Overfitting / underfitting detection, convergence analysis, and hyperparameter recommendations")

    architecture = st.text_area(
        "Model architecture description",
        placeholder="e.g. 3-layer MLP, 256-128-10, ReLU, Dropout 0.1, Adam lr=0.001...",
        height=120,
    )
    training_logs = st.text_area(
        "Training logs",
        placeholder="Paste epoch-wise train/val loss or full training log...",
        height=120,
    )
    col1, col2 = st.columns(2)
    with col1:
        train_loss_str = st.text_input(
            "Train loss (comma- or space-separated)",
            placeholder="0.89, 0.65, 0.41, 0.20, 0.08, 0.02",
        )
    with col2:
        val_loss_str = st.text_input(
            "Val loss (comma- or space-separated)",
            placeholder="0.85, 0.62, 0.52, 0.49, 0.51, 0.56",
        )

    if st.button("Run analysis", type="primary", use_container_width=True):
        if not train_loss_str.strip() or not val_loss_str.strip():
            st.error("Provide both train loss and val loss.")
            return
        try:
            train_loss = _parse_floats(train_loss_str)
            val_loss = _parse_floats(val_loss_str)
        except ValueError as e:
            st.error(f"Invalid loss values: {e}")
            return
        if len(train_loss) != len(val_loss):
            st.error("Train loss and val loss must have the same number of values.")
            return
        if not train_loss:
            st.error("At least one loss value required.")
            return

        input_data = ModelDebugInput(
            architecture=architecture.strip() or "(none provided)",
            training_logs=training_logs.strip() or "(none provided)",
            train_loss=train_loss,
            val_loss=val_loss,
        )

        with st.spinner("Running deterministic analysis and LLM..."):
            try:
                result = run_debug_agent(input_data)
            except Exception as e:
                st.error(str(e))
                return

        analysis = result.pop("_deterministic_analysis", None)
        if analysis:
            with st.expander("Deterministic analysis", expanded=True):
                st.json(analysis)

        diagnosis = result.get("training_diagnosis", {})
        if diagnosis:
            st.subheader("Training diagnosis")
            c1, c2, c3 = st.columns(3)
            c1.metric("Overfitting", "Yes" if diagnosis.get("overfitting") else "No")
            c2.metric("Underfitting", "Yes" if diagnosis.get("underfitting") else "No")
            c3.metric("Convergence", diagnosis.get("convergence_quality", ""))

        st.subheader("Recommendations")
        for key, label in [
            ("hyperparameter_recommendations", "Hyperparameter recommendations"),
            ("architecture_improvements", "Architecture improvements"),
            ("regularization_strategies", "Regularization strategies"),
            ("optimization_strategies", "Optimization strategies"),
            ("data_recommendations", "Data recommendations"),
            ("priority_action_plan", "Priority action plan"),
        ]:
            items = result.get(key)
            if items:
                with st.expander(label):
                    if isinstance(items, list) and items and isinstance(items[0], dict):
                        for i, row in enumerate(items, 1):
                            param = row.get("parameter", "")
                            st.markdown(f"**{i}. {param}**")
                            if row.get("suggestion"):
                                st.write(row["suggestion"])
                            if row.get("recommended_range"):
                                st.caption(f"Range: {row['recommended_range']}")
                    else:
                        for i, item in enumerate(items, 1):
                            st.markdown(f"{i}. {item}")

        with st.expander("Full JSON output"):
            st.json(result)

        if train_loss and val_loss:
            try:
                curve_path = plot_loss_curves(train_loss, val_loss)
                st.subheader("Loss curves")
                st.image(str(curve_path), use_container_width=True)
            except Exception as e:
                logger.warning("Could not plot loss curve: %s", e)


def cli() -> None:
    """Minimal CLI: paste architecture, logs, train loss, val loss; print JSON."""
    print("Paste model architecture description (end with empty line or EOF):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    architecture = "\n".join(lines) if lines else ""

    print("Paste training logs (end with empty line or EOF):")
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if not line.strip():
            break
        lines.append(line)
    training_logs = "\n".join(lines) if lines else ""

    print("Paste comma- or space-separated train loss values:")
    train_line = input().strip()
    train_loss = _parse_floats(train_line)

    print("Paste comma- or space-separated val loss values:")
    val_line = input().strip()
    val_loss = _parse_floats(val_line)

    if len(train_loss) != len(val_loss):
        print("Error: train_loss and val_loss must have the same length", file=sys.stderr)
        sys.exit(1)
    if not train_loss:
        print("Error: at least one loss value required", file=sys.stderr)
        sys.exit(1)

    input_data = ModelDebugInput(
        architecture=architecture or "(none provided)",
        training_logs=training_logs or "(none provided)",
        train_loss=train_loss,
        val_loss=val_loss,
    )

    result = run_debug_agent(input_data)
    analysis = result.pop("_deterministic_analysis", None)

    if analysis and (train_loss or val_loss):
        try:
            curve_path = plot_loss_curves(train_loss, val_loss)
            print(f"Loss curve saved: {curve_path}", file=sys.stderr)
        except Exception as e:
            logger.warning("Could not save loss curve: %s", e)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    if "streamlit" in sys.modules:
        run_streamlit_app()
    else:
        cli()
