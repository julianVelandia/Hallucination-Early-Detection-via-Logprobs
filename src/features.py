"""
Feature engineering module — Tarea 3.

Computes a rich numerical feature vector from the raw logprob sequences
stored in results.json, then merges with the labels from labeled_dataset.csv
to produce the final feature matrix ready for ML training.

Feature groups
--------------
Base stats (already in CSV, re-used here):
  logprob_n_tokens, logprob_mean, logprob_std, logprob_min, logprob_max,
  logprob_p10, logprob_p90, logprob_mean_top_entropy

Extended sequence features (computed from raw JSON logprobs):
  consec_drop_max   — largest single-step decrease between consecutive tokens
  consec_drop_mean  — mean of all consecutive drops
  consec_drop_std   — std of consecutive drops
  frac_below_m1     — fraction of tokens with logprob < -1.0
  frac_below_m2     — fraction of tokens with logprob < -2.0
  margin_mean       — mean(top1_logprob - top2_logprob) per token (confidence gap)
  margin_std        — std of that gap
  margin_min        — worst-case confidence gap (most uncertain single token)
"""

import json
import math
import statistics
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------------
# Extended features from a raw logprob sequence
# ---------------------------------------------------------------------------

def compute_extended_features(logprobs: list) -> dict:
    """
    Derive sequence-level features from a list of per-token logprob dicts.

    Each dict: {"token": str, "logprob": float, "top_logprobs": [...]}
    """
    values = [lp["logprob"] for lp in logprobs if lp.get("logprob") is not None]
    n = len(values)

    if n == 0:
        return {k: None for k in [
            "consec_drop_max", "consec_drop_mean", "consec_drop_std",
            "frac_below_m1", "frac_below_m2",
            "margin_mean", "margin_std", "margin_min",
        ]}

    # Consecutive drops: lp[i] - lp[i+1]  (positive = drop in confidence)
    drops = [values[i] - values[i + 1] for i in range(n - 1)]

    # Confidence margin: top1 - top2 logprob at each position
    margins = []
    for lp in logprobs:
        tops = lp.get("top_logprobs") or []
        if len(tops) >= 2:
            margin = tops[0]["logprob"] - tops[1]["logprob"]
            margins.append(margin)

    def _safe(fn, data, default=None):
        try:
            return round(fn(data), 6) if data else default
        except Exception:
            return default

    return {
        "consec_drop_max":  _safe(max, drops),
        "consec_drop_mean": _safe(statistics.mean, drops),
        "consec_drop_std":  _safe(statistics.stdev, drops) if len(drops) > 1 else 0.0,
        "frac_below_m1":    round(sum(1 for v in values if v < -1.0) / n, 6),
        "frac_below_m2":    round(sum(1 for v in values if v < -2.0) / n, 6),
        "margin_mean":      _safe(statistics.mean, margins),
        "margin_std":       _safe(statistics.stdev, margins) if len(margins) > 1 else 0.0,
        "margin_min":       _safe(min, margins),
    }


# ---------------------------------------------------------------------------
# Build the full feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    csv_path: str,
    results_json_path: str,
) -> pd.DataFrame:
    """
    Load the labeled CSV, enrich each row with extended features from the raw
    JSON logprobs, and return the complete feature DataFrame.

    The source files are treated as READ-ONLY; no data is written here.

    Parameters
    ----------
    csv_path : str
        Path to labeled_dataset.csv (output of Tarea 2).
    results_json_path : str
        Path to results.json (output of Tarea 1, contains raw logprob sequences).

    Returns
    -------
    pd.DataFrame with all base + extended features and the target column
    ``hallucination_score``.
    """
    # --- load labeled CSV ---
    df = pd.read_csv(csv_path)

    # --- load raw logprobs ---
    with open(results_json_path, encoding="utf-8") as f:
        raw = json.load(f)
    processed = raw.get("processed", {})

    # Index by (question_id, model, turn) for quick lookup
    logprob_index: dict = {}
    for entry in processed.values():
        q_id = entry["question_id"]
        model = entry["model"]
        for t in entry.get("turns", []):
            logprob_index[(q_id, model, t["turn"])] = t.get("logprobs", [])

    # --- compute extended features per row ---
    ext_records = []
    for _, row in df.iterrows():
        key = (row["question_id"], row["model"], int(row["turn"]))
        lp_seq = logprob_index.get(key, [])
        ext = compute_extended_features(lp_seq)
        ext_records.append(ext)

    ext_df = pd.DataFrame(ext_records)
    result = pd.concat([df.reset_index(drop=True), ext_df], axis=1)
    return result


# ---------------------------------------------------------------------------
# Feature column lists for downstream use
# ---------------------------------------------------------------------------

BASE_FEATURE_COLS = [
    "logprob_n_tokens",
    "logprob_mean",
    "logprob_std",
    "logprob_min",
    "logprob_max",
    "logprob_p10",
    "logprob_p90",
    "logprob_mean_top_entropy",
]

EXTENDED_FEATURE_COLS = [
    "consec_drop_max",
    "consec_drop_mean",
    "consec_drop_std",
    "frac_below_m1",
    "frac_below_m2",
    "margin_mean",
    "margin_std",
    "margin_min",
]

ALL_FEATURE_COLS = BASE_FEATURE_COLS + EXTENDED_FEATURE_COLS

TARGET_COL = "hallucination_score"
