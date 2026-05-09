# Hallucination Early Detection via Logprobs


> **"Mitigating Hallucination Propagation in Sequential Reasoning of Autonomous Agents through Early Detection and Adaptive Resolution"**
> Julián Camilo Velandia Gutiérrez — Universidad Nacional de Colombia, 2026
> Advisor: Ph.D. Luis Fernando Niño Vásquez · Research Group: LISI

---

## Research Problem

When an autonomous agent hallucinates in the **first step** of a sequential reasoning chain, the false premise gets written into its context window. Every subsequent step builds on that error, producing a degenerative loop that cannot be broken without external intervention. The model keeps executing code and refining logic derived from a false observation, while the execution environment returns no exceptions — a **silent failure**.

This repository implements the experimental pipeline that tests whether **token-level log-probabilities** (logprobs) from the model's own output carry a detectable signal that reliably distinguishes a hallucinated response from a correct one — before the error propagates.

---

## Approach

The proposed method operates in three stages:

```
Agent response + logprobs
        │
        ▼
  Feature extraction          ← statistical summaries of the logprob sequence
        │
        ▼
  Lightweight classifier       ← LightGBM / XGBoost / Logistic Regression
        │
        ▼
  Hallucination score [0,1]   ← triggers adaptive resolution if above threshold
```

The classifier adds negligible latency (~0.013 ms per inference) to the agent loop, making it viable for real-time deployment alongside LLM agents.

---

## Pipeline Overview

| Stage | Task | Command |
|-------|------|---------|
| **1 — Data Collection** | Run GPT-4o-mini on the Bullshit Benchmark; capture full token logprobs per response turn | `python cli.py run` |
| **2 — Labeling** | LLM judge (GPT-4o) scores each response: `0.0` correct · `0.5` ambiguous · `1.0` hallucination | `python cli.py label run` |
| **3 — ML Pipeline** | Feature engineering from logprob sequences → train classifiers → evaluate → benchmark latency | `python cli.py train run` |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
cp .env.example .env   # then add OPENAI_API_KEY=sk-...

# 3. Explore the benchmark dataset
python cli.py explore

# 4. Collect interaction traces (10 questions, 1 turn each)
python cli.py run --questions 10 --turns 1

# 5. Label hallucinations with the LLM judge
python cli.py label run

# 6. Train and evaluate the predictive models
python cli.py train run

# 7. Check the report
python cli.py train report
```

---

## Commands Reference

### Data Collection

```bash
python cli.py run --questions N          # collect N questions (0 = all)
python cli.py run --questions 20 --turns 2 --domain legal
python cli.py run --dry-run              # preview without API calls
python cli.py status                     # collection progress
python cli.py explore                    # dataset statistics
python cli.py reset --question-id ID     # re-collect a specific question
```

### Hallucination Labeling

```bash
python cli.py label run                          # label all unlabeled turns
python cli.py label run --labeler-model gpt-4o-mini --limit 20
python cli.py label run --no-export              # skip CSV/Excel export
python cli.py label export --format csv          # re-export only
python cli.py label status                       # labeling progress
```

### ML Training & Evaluation

```bash
python cli.py train run                          # full pipeline
python cli.py train run --cv-folds 10 --features all
python cli.py train run --features base          # base logprob stats only
python cli.py train report                       # display last report
python cli.py train benchmark --runs 1000        # re-run latency benchmark
```

### Configuration

```bash
python cli.py config show
python cli.py config set --model gpt-4o-mini --turns 2
python cli.py config set --top-logprobs 5
```

---

## Features Extracted from Logprob Sequences

| Feature | Description |
|---------|-------------|
| `logprob_mean` | Mean log-probability across all output tokens |
| `logprob_std` | Standard deviation (spread of uncertainty) |
| `logprob_min / max` | Worst- and best-case token confidence |
| `logprob_p10 / p90` | 10th and 90th percentile |
| `logprob_mean_top_entropy` | Mean entropy of top-k alternatives per token |
| `consec_drop_max` | Largest single-step confidence drop between consecutive tokens |
| `consec_drop_mean / std` | Distribution of consecutive drops |
| `frac_below_m1 / m2` | Fraction of tokens with logprob < −1.0 / −2.0 |
| `margin_mean / std / min` | Gap between best and second-best token at each position |

---

## Labeling Scale

| Score | Meaning |
|-------|---------|
| `0.0` | **No hallucination** — agent correctly identified the false premise |
| `0.5` | **Partial / ambiguous** — hedged response, neither confirms nor refutes |
| `1.0` | **Full hallucination** — agent accepted the false premise and invented supporting reasoning |

---

## Models

Three classifiers are trained and compared for suitability in the agent loop:

- **LightGBM** _(recommended)_ — fastest inference, lowest memory footprint
- **XGBoost** — strong tabular performance, built-in feature importance
- **Logistic Regression** — lightest possible model; baseline for interpretability

All models are saved as `.pkl` files in `models/` and can be loaded independently for inference.

---

## Output Files

| File | Description |
|------|-------------|
| `data/results.json` | Raw interaction traces + full token logprobs (Tarea 1) |
| `data/labels.json` | Incremental hallucination labels (Tarea 2) |
| `data/labeled_dataset.csv` | Flat labeled dataset ready for ML ingestion |
| `data/labeled_dataset.xlsx` | Same, Excel format |
| `data/features.csv` | Full feature matrix (base + extended) |
| `data/ml_report.txt` | Human-readable metrics + latency report (Tarea 3) |
| `data/ml_report.csv` | Machine-readable metrics table |
| `data/confusion_matrices.png` | Confusion matrices per model |
| `data/roc_curves.png` | ROC curves (generated when ≥ 2 classes present) |
| `models/*.pkl` | Trained model artifacts |

---

## Benchmark Dataset

Data collection uses the [Bullshit Benchmark](https://github.com/petergpt/bullshit-benchmark), a dataset of questions that embed plausible but completely fabricated concepts across domains (legal, medical, software engineering). The benchmark is designed to test whether a model validates false premises rather than rejecting them — exactly the failure mode this research targets.

---

## Requirements

- Python 3.10+
- OpenAI API key with access to `gpt-4o-mini` (collection) and `gpt-4o` (labeling)
- See `requirements.txt` for full dependency list

---

## Project Structure

```
poc-2/
├── cli.py                  # Main CLI entry point
├── src/
│   ├── dataset.py          # Dataset fetching and parsing
│   ├── runner.py           # Agent interaction loop (logprob capture)
│   ├── storage.py          # JSON result persistence
│   ├── labeler.py          # LLM judge labeling pipeline
│   ├── features.py         # Feature engineering from logprob sequences
│   └── ml_pipeline.py      # ML training, evaluation, and benchmarking
├── data/                   # Generated datasets and reports
├── models/                 # Saved model artifacts
├── config.json             # Runtime configuration
└── requirements.txt
```

---

## Citation

If you use this code, please cite the associated thesis:

```
Velandia Gutiérrez, J. C. (2026). Mitigating Hallucination Propagation in Sequential
Reasoning of Autonomous Agents through Early Detection and Adaptive Resolution.
 Universidad Nacional de Colombia, Faculty of Engineering.
```
