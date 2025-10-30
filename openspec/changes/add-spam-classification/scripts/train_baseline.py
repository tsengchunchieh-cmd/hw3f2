"""Train a baseline spam classifier.

By default this script trains an SVM baseline on the provided dataset and writes
model, vectorizer, and metrics to an artifacts/ directory.

Usage examples:
python scripts/train_baseline.py --data data/sms_spam_no_header.csv --model svm
python scripts/train_baseline.py --data <url-or-path> --model logreg --out artifacts
"""
import os
import argparse
import json
import re
from pprint import pprint
import sys

import joblib
import numpy as np
import pandas as pd
from pprint import pprint

# Heavy ML imports (scikit-learn) are done lazily inside train() to improve
# diagnostics and avoid import-time hangs in some environments.


DEFAULT_URL = (
    "https://raw.github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
)

RANDOM_STATE = 42


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_data(path_or_url: str):
    # If path_or_url looks like a URL, pass to pandas directly; else read file
    try:
        df = pd.read_csv(path_or_url, header=None, names=["label", "message"], encoding="utf-8", engine="python")
    except Exception:
        df = pd.read_csv(path_or_url, header=None, names=["label", "message"], encoding="latin-1", engine="python")
    # Basic cleaning
    df = df.dropna().reset_index(drop=True)
    df["message_clean"] = df["message"].apply(clean_text)
    df["label_bin"] = df["label"].apply(lambda x: 1 if str(x).strip().lower() in ("spam", "1", "true") else 0)
    return df


def train(df: pd.DataFrame, model_type: str = "svm", out_dir: str = "artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    X = df["message_clean"].values
    y = df["label_bin"].values

    # Import ML libraries here to allow earlier parts of the script to run
    # and to provide clearer error messages if imports fail.
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.svm import LinearSVC
        from sklearn.linear_model import LogisticRegression
    except Exception as e:
        print("Failed to import scikit-learn or its dependencies:", repr(e))
        print("Traceback will follow. Consider running the diagnostic script 'scripts/diagnose_env.py' to collect environment info.")
        # Exit cleanly so we don't hit UnboundLocalError later
        sys.exit(1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y if len(np.unique(y)) > 1 else None
    )

    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    if model_type == "svm":
        model = LinearSVC(random_state=RANDOM_STATE)
    elif model_type == "logreg":
        model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    else:
        raise ValueError("Unknown model type: choose 'svm' or 'logreg'")

    model.fit(X_train_tfidf, y_train)
    preds = model.predict(X_test_tfidf)

    report = classification_report(y_test, preds, output_dict=True)
    cm = confusion_matrix(y_test, preds).tolist()

    metrics = {
        "classification_report": report,
        "confusion_matrix": cm,
        "model_type": model_type,
    }

    # Save artifacts
    model_path = os.path.join(out_dir, f"model_{model_type}.joblib")
    vec_path = os.path.join(out_dir, f"vectorizer_{model_type}.joblib")
    metrics_path = os.path.join(out_dir, f"metrics_{model_type}.json")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print("Training complete. Artifacts written to:")
    print(f"  model: {model_path}")
    print(f"  vectorizer: {vec_path}")
    print(f"  metrics: {metrics_path}")
    pprint(metrics)

    return model_path, vec_path, metrics_path


def compare_models(df: pd.DataFrame, out_dir: str = "artifacts", models=("svm", "logreg")):
    """Train multiple models (SVM and LogReg), collect their metrics, and write a unified comparison JSON.

    The function calls `train()` for each model which will persist model and vectorizer
    files into `out_dir`. It then reads the per-model metrics files and writes
    `comparison.json` under `out_dir` containing both metrics and a simple summary.
    """
    os.makedirs(out_dir, exist_ok=True)
    comparison = {"models": {}, "summary": {}}

    for m in models:
        print(f"Training and evaluating model: {m}")
        model_path, vec_path, metrics_path = train(df, model_type=m, out_dir=out_dir)
        # load metrics
        try:
            with open(metrics_path, "r", encoding="utf-8") as fh:
                metrics = json.load(fh)
        except Exception:
            metrics = None
        comparison["models"][m] = {
            "model_path": model_path,
            "vectorizer_path": vec_path,
            "metrics": metrics,
        }

    # simple comparative summary: accuracy per model
    summary = {}
    for m, info in comparison["models"].items():
        metrics = info.get("metrics") or {}
        acc = None
        # classification_report may be nested; try to extract accuracy
        if isinstance(metrics.get("classification_report"), dict):
            acc = metrics["classification_report"].get("accuracy")
        summary[m] = {"accuracy": acc}

    comparison["summary"] = summary

    comp_path = os.path.join(out_dir, "comparison.json")
    with open(comp_path, "w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2)

    print(f"Wrote comparison report to: {comp_path}")
    return comp_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_URL, help="Path or URL to dataset CSV (no header, label,message)")
    parser.add_argument("--model", choices=("svm", "logreg"), default="svm", help="Model type: svm (default) or logreg")
    parser.add_argument("--out", default="artifacts", help="Output directory for artifacts")
    args = parser.parse_args()
    df = load_data(args.data)
    train(df, model_type=args.model, out_dir=args.out)
