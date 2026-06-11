# agentic-compliance-probe

Research-grade evaluation harness for measuring instruction compliance, tool-use correctness, reasoning-action consistency, and evaluation-aware behavior shifts across free-tier LLM APIs.

## What this project does

This is not a chatbot. It is a reproducible experiment pipeline that:

- loads fixed evaluation scenarios
- queries Groq-hosted LLaMA or Mixtral models and Google Gemini Flash models
- scores each response with deterministic heuristics
- writes structured JSON logs for analysis
- summarizes results in a notebook

## Project layout

- `tools/mock_tools.py` contains fake tools that return structured JSON.
- `evaluation/compliance_checker.py` computes compliance scores.
- `scenarios/scenarios.json` defines the benchmark scenarios.
- `pipeline/run_experiments.py` orchestrates model runs and logging.
- `results/` stores JSONL outputs from experiment runs.
- `analysis/analysis.ipynb` compares model behavior and highlights failures.

## Setup

1. Create a Python environment.
2. Install dependencies from `requirements.txt`.
3. Set environment variables for your API keys:

```bash
GROQ_API_KEY=...
GOOGLE_API_KEY=...
```

Optional: place the same values in `.env` at the repository root.

## Install

```bash
pip install -r requirements.txt
```

## Run experiments

```bash
python pipeline/run_experiments.py
```

You can also target a subset of models:

```bash
python pipeline/run_experiments.py --models groq:llama-3.3-70b-versatile gemini:gemini-1.5-flash
```

Each run writes a timestamped JSONL file into `results/` and updates `results/latest_run.json`.

## Analysis

Open `analysis/analysis.ipynb` after running experiments. The notebook computes:

- average compliance by model and condition
- bar charts of compliance scores
- failure-case tables
- reasoning-action mismatch rates

## Reproducibility notes

- Temperature is fixed at `0.0`.
- Scenarios are stored in versioned JSON.
- Results are written as JSONL for easy reprocessing.
- The scoring function is deterministic and local.

## Safety note

The mock tools do not execute real side effects. They only return structured responses so the benchmark can study model behavior without external operations.
