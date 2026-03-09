# Model Debug Agent

Part of [Anuj-AI-ML-Lab](https://github.com/Anujpatel04/Anuj-AI-ML-Lab/tree/main).

Analyzes model training: overfitting/underfitting detection, convergence, and hyperparameter/architecture recommendations. Uses deterministic analysis plus DeepSeek (LLM) for structured suggestions.

## Requirements

- Python 3.9+
- `DEEPSEEK_API_KEY` in the project root `.env` (parent of `AI_AGENTS`)

## Installation

```bash
cd AI_AGENTS/Model_Evaluation_Agent
pip install -r requirements.txt
```

## Usage

**Streamlit UI**

```bash
streamlit run model_debug_agent.py
```

**CLI**

```bash
python model_debug_agent.py
```

Enter architecture, training logs, and comma- or space-separated train/val loss when prompted.

## Input

- **Architecture:** Model description (layers, optimizer, etc.).
- **Training logs:** Epoch-wise or full training log.
- **Train loss / Val loss:** Same-length arrays of per-epoch values.

## Output

JSON with: `training_diagnosis`, `hyperparameter_recommendations`, `architecture_improvements`, `regularization_strategies`, `optimization_strategies`, `data_recommendations`, `priority_action_plan`. Optionally plots train/val loss curve as PNG.

## Configuration

- **DEEPSEEK_API_KEY:** Required.
- **DEEPSEEK_MODEL:** Optional; default `deepseek-chat`.
- **DEEPSEEK_BASE_URL:** Optional; default `https://api.deepseek.com/v1`.
