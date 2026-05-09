"""
ML Pipeline module — Tarea 3.

Trains LightGBM, XGBoost, and Logistic Regression classifiers on the feature
matrix produced by src/features.py, evaluates them, benchmarks inference
latency, and exports all artefacts.

Design notes
------------
- Target encoding: 0 = no/partial error (score 0.0 or 0.5),
                   1 = full hallucination (score 1.0).
- Requires at least MIN_SAMPLES_PER_CLASS samples in EACH class for training.
  If the condition is not met (e.g. early in data collection), the pipeline
  reports the limitation and skips cross-validation.
- All source files (CSV, JSON) are opened READ-ONLY inside this module.
"""

import json
import pickle
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

MIN_SAMPLES_PER_CLASS = 3   # minimum samples per class to run CV


# ---------------------------------------------------------------------------
# Target binarization
# ---------------------------------------------------------------------------

def binarize_target(series: pd.Series) -> pd.Series:
    """Map 0.0/0.5 → 0  and  1.0 → 1."""
    return (series >= 1.0).astype(int)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def _build_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    models = {}

    try:
        import lightgbm as lgb
        models["LightGBM"] = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            num_leaves=15,
            random_state=42,
            verbose=-1,
        )
    except ImportError:
        pass

    try:
        import xgboost as xgb
        models["XGBoost"] = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42,
            verbosity=0,
        )
    except ImportError:
        pass

    models["LogisticRegression"] = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
            solver="lbfgs",
        )),
    ])

    return models


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate_model(model, X, y, cv_folds: int):
    """
    Run stratified K-fold CV and return a metrics dict.

    Returns a dict with keys:
      accuracy, precision, recall, f1, auc  (all mean ± std from CV)
      plus raw per-fold arrays for reporting.
    """
    from sklearn.model_selection import StratifiedKFold, cross_validate
    from sklearn.metrics import make_scorer, roc_auc_score, f1_score

    scorers = {
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "f1": "f1_weighted",
    }

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_results = cross_validate(
        model, X, y,
        cv=skf,
        scoring=scorers,
        return_train_score=False,
        error_score="raise",
    )

    # AUC requires predict_proba; fall back gracefully
    auc_scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        model.fit(X_tr, y_tr)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_te)
            if proba.shape[1] >= 2:
                try:
                    auc_scores.append(roc_auc_score(y_te, proba[:, 1]))
                except Exception:
                    pass
        elif hasattr(model, "decision_function"):
            try:
                auc_scores.append(roc_auc_score(y_te, model.decision_function(X_te)))
            except Exception:
                pass

    metrics = {}
    for key in scorers:
        arr = cv_results[f"test_{key}"]
        metrics[f"{key}_mean"] = round(float(np.mean(arr)), 4)
        metrics[f"{key}_std"] = round(float(np.std(arr)), 4)

    metrics["auc_mean"] = round(float(np.mean(auc_scores)), 4) if auc_scores else None
    metrics["auc_std"] = round(float(np.std(auc_scores)), 4) if auc_scores else None
    return metrics


def _confusion_matrix_dict(model, X, y):
    from sklearn.metrics import confusion_matrix
    model.fit(X, y)
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    labels = sorted(set(y))
    return {"matrix": cm.tolist(), "labels": labels}


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

def benchmark_latency(model, X: np.ndarray, n_runs: int = 200) -> dict:
    """
    Measure per-sample inference latency by passing one row at a time.

    Returns a dict with p50, p95, p99, mean, std (all in milliseconds).
    """
    times = []
    for _ in range(n_runs):
        idx = np.random.randint(0, len(X))
        sample = X[idx : idx + 1]
        t0 = time.perf_counter()
        model.predict(sample)
        times.append((time.perf_counter() - t0) * 1000)

    arr = np.array(times)
    return {
        "mean_ms": round(float(np.mean(arr)), 4),
        "std_ms": round(float(np.std(arr)), 4),
        "p50_ms": round(float(np.percentile(arr, 50)), 4),
        "p95_ms": round(float(np.percentile(arr, 95)), 4),
        "p99_ms": round(float(np.percentile(arr, 99)), 4),
    }


# ---------------------------------------------------------------------------
# ROC / Confusion matrix plots (optional, requires matplotlib)
# ---------------------------------------------------------------------------

def _save_plots(results: dict, X: np.ndarray, y: np.ndarray, output_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import (
            RocCurveDisplay, ConfusionMatrixDisplay, confusion_matrix
        )
    except ImportError:
        return []

    saved = []
    n_models = len(results)

    # --- Confusion matrices ---
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4))
    if n_models == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        cm_data = res.get("confusion_matrix", {})
        cm = np.array(cm_data.get("matrix", [[0]]))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=cm_data.get("labels", [0, 1]),
        )
        disp.plot(ax=ax, colorbar=False)
        ax.set_title(name)
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrices.png"
    fig.savefig(cm_path, dpi=120)
    plt.close(fig)
    saved.append(str(cm_path))

    # --- ROC curves ---
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, res in results.items():
        model = res.get("_model")
        if model is None:
            continue
        if not hasattr(model, "predict_proba"):
            continue
        try:
            RocCurveDisplay.from_estimator(model, X, y, ax=ax, name=name)
        except Exception:
            pass
    ax.set_title("ROC Curves (train set)")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    plt.tight_layout()
    roc_path = output_dir / "roc_curves.png"
    fig.savefig(roc_path, dpi=120)
    plt.close(fig)
    saved.append(str(roc_path))

    return saved


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    feature_df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    models_dir: str,
    output_dir: str,
    cv_folds: int = 5,
    benchmark_runs: int = 200,
) -> dict:
    """
    Full training, evaluation, and benchmarking pipeline.

    Parameters
    ----------
    feature_df    : DataFrame with features + target (READ-ONLY — not modified).
    feature_cols  : List of column names to use as features.
    target_col    : Column with the hallucination score (continuous 0/0.5/1.0).
    models_dir    : Directory where .pkl model files are saved.
    output_dir    : Directory for report files and plots.
    cv_folds      : K for StratifiedKFold cross-validation.
    benchmark_runs: Number of single-sample passes for latency measurement.

    Returns
    -------
    dict with full results: metrics per model, latency, confusion matrices, paths.
    """
    from sklearn.impute import SimpleImputer

    models_path = Path(models_dir)
    output_path = Path(output_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    output_path.mkdir(parents=True, exist_ok=True)

    # --- Prepare X, y ---
    df = feature_df.copy()
    y_raw = df[target_col]
    y = binarize_target(y_raw).values

    X_df = df[feature_cols].copy()
    imputer = SimpleImputer(strategy="mean")
    X = imputer.fit_transform(X_df).astype(np.float32)

    # Save imputer
    with open(models_path / "imputer.pkl", "wb") as f:
        pickle.dump({"imputer": imputer, "feature_cols": feature_cols}, f)

    # --- Class distribution check ---
    classes, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(classes.tolist(), counts.tolist()))
    can_train_cv = (
        len(classes) >= 2
        and all(c >= MIN_SAMPLES_PER_CLASS for c in counts)
    )

    pipeline_summary = {
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "class_distribution": class_dist,
        "cv_folds": cv_folds,
        "can_train_cv": can_train_cv,
        "warning": None if can_train_cv else (
            f"Insufficient class diversity for CV. "
            f"Need >= {MIN_SAMPLES_PER_CLASS} samples per class. "
            f"Current: {class_dist}. Collect more data with class 0 (score 0.0/0.5)."
        ),
    }

    models = _build_models()
    results = {}

    for name, model in models.items():
        model_result = {"name": name}

        if can_train_cv:
            # Full CV evaluation
            cv_metrics = _evaluate_model(model, X, y, cv_folds)
            model_result.update(cv_metrics)
            cm_info = _confusion_matrix_dict(model, X, y)
            model_result["confusion_matrix"] = cm_info
        else:
            # Degenerate case: use DummyClassifier (predicts majority class)
            # — real models require at least 2 classes to fit properly.
            from sklearn.dummy import DummyClassifier
            from sklearn.metrics import (
                accuracy_score, f1_score, recall_score, precision_score,
                confusion_matrix as cm_fn,
            )
            dummy = DummyClassifier(strategy="most_frequent")
            dummy.fit(X, y)
            y_pred = dummy.predict(X)
            model_result["note"] = (
                "DummyClassifier (most_frequent) used — insufficient class "
                "diversity for real model training. Collect data with score 0.0/0.5."
            )
            model_result["accuracy_mean"] = round(float(accuracy_score(y, y_pred)), 4)
            model_result["f1_mean"] = round(float(f1_score(y, y_pred, average="weighted", zero_division=0)), 4)
            model_result["recall_mean"] = round(float(recall_score(y, y_pred, average="weighted", zero_division=0)), 4)
            model_result["precision_mean"] = round(float(precision_score(y, y_pred, average="weighted", zero_division=0)), 4)
            model_result["confusion_matrix"] = {
                "matrix": cm_fn(y, y_pred).tolist(),
                "labels": sorted(set(y.tolist())),
            }
            # Use dummy for latency benchmark too (real model can't be trained)
            model = dummy

        # Final fit on full data for latency benchmark + saving
        try:
            model.fit(X, y)
        except Exception:
            pass  # already fitted (dummy) or already evaluated via CV
        latency = benchmark_latency(model, X, n_runs=benchmark_runs)
        model_result["latency"] = latency

        # Save model
        pkl_path = models_path / f"{name.lower().replace(' ', '_')}.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(model, f)
        model_result["model_path"] = str(pkl_path)
        model_result["_model"] = model  # kept for plotting, not serialized

        results[name] = model_result

    # --- Plots ---
    plot_paths = []
    if can_train_cv:
        plot_paths = _save_plots(results, X, y, output_path)

    # --- Feature importances (tree models) ---
    importances = {}
    for name, model in models.items():
        clf = model[-1] if hasattr(model, "__getitem__") else model  # unwrap Pipeline
        if hasattr(clf, "feature_importances_"):
            imp = dict(zip(feature_cols, clf.feature_importances_.tolist()))
            imp_sorted = dict(sorted(imp.items(), key=lambda x: -x[1]))
            importances[name] = imp_sorted

    return {
        "summary": pipeline_summary,
        "models": {k: {kk: vv for kk, vv in v.items() if kk != "_model"}
                   for k, v in results.items()},
        "feature_importances": importances,
        "plots": plot_paths,
        "imputer_path": str(models_path / "imputer.pkl"),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(pipeline_result: dict, output_path: str) -> str:
    """
    Write a human-readable text report and a CSV metrics table.

    Returns the path to the text report.
    """
    txt_path = Path(output_path).with_suffix(".txt")
    csv_path = Path(output_path).with_suffix(".csv")
    output_dir = txt_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pipeline_result["summary"]
    models = pipeline_result["models"]
    importances = pipeline_result.get("feature_importances", {})

    # --- Text report ---
    lines = [
        "=" * 70,
        "HALLUCINATION DETECTION — ML PIPELINE REPORT",
        "Tarea 3: Pipeline Predictivo",
        "=" * 70,
        "",
        f"Samples         : {summary['n_samples']}",
        f"Features        : {summary['n_features']}",
        f"Class distribution: {summary['class_distribution']}",
        f"CV folds        : {summary['cv_folds']}",
    ]
    if summary.get("warning"):
        lines += ["", f"[WARNING] {summary['warning']}", ""]

    lines += ["", "-" * 70, "CLASSIFICATION METRICS (mean over CV folds)", "-" * 70]

    metric_keys = ["accuracy_mean", "precision_mean", "recall_mean", "f1_mean", "auc_mean"]
    header = f"{'Model':<22}" + "".join(f"{m.replace('_mean',''):>12}" for m in metric_keys)
    lines.append(header)
    lines.append("-" * len(header))

    csv_rows = []
    for name, res in models.items():
        row_str = f"{name:<22}"
        csv_row = {"model": name}
        for mk in metric_keys:
            val = res.get(mk)
            row_str += f"{'N/A':>12}" if val is None else f"{val:>12.4f}"
            csv_row[mk.replace("_mean", "")] = val
        # std
        for mk in ["accuracy_std", "precision_std", "recall_std", "f1_std", "auc_std"]:
            csv_row[mk] = res.get(mk)
        lines.append(row_str)
        csv_rows.append(csv_row)

    lines += ["", "-" * 70, "INFERENCE LATENCY (ms, 200 single-sample passes)", "-" * 70]
    lat_header = f"{'Model':<22}{'mean':>10}{'std':>10}{'p50':>10}{'p95':>10}{'p99':>10}"
    lines.append(lat_header)
    lines.append("-" * len(lat_header))

    for name, res in models.items():
        lat = res.get("latency", {})
        lines.append(
            f"{name:<22}"
            f"{lat.get('mean_ms', 0):>10.4f}"
            f"{lat.get('std_ms', 0):>10.4f}"
            f"{lat.get('p50_ms', 0):>10.4f}"
            f"{lat.get('p95_ms', 0):>10.4f}"
            f"{lat.get('p99_ms', 0):>10.4f}"
        )
        csv_rows_dict = next(r for r in csv_rows if r["model"] == name)
        for k, v in lat.items():
            csv_rows_dict[k] = v

    if importances:
        lines += ["", "-" * 70, "FEATURE IMPORTANCES (top 8)", "-" * 70]
        for name, imp in importances.items():
            lines.append(f"\n{name}:")
            for feat, score in list(imp.items())[:8]:
                bar = "#" * int(score * 40)
                lines.append(f"  {feat:<35} {score:.4f}  {bar}")

    if pipeline_result.get("plots"):
        lines += ["", "-" * 70, "GENERATED PLOTS", "-" * 70]
        for p in pipeline_result["plots"]:
            lines.append(f"  {p}")

    lines += ["", "=" * 70]
    report_text = "\n".join(lines)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # --- CSV metrics ---
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)

    return str(txt_path)
