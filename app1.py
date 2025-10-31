import os
import argparse
import json
import re
import sys
import time
import platform
import subprocess
from pprint import pprint

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --- Common functions ---
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
    try:
        df = pd.read_csv(path_or_url, header=None, names=["label", "message"], encoding="utf-8", engine="python")
    except Exception:
        df = pd.read_csv(path_or_url, header=None, names=["label", "message"], encoding="latin-1", engine="python")
    df = df.dropna().reset_index(drop=True)
    df["message_clean"] = df["message"].apply(clean_text)
    df["label_bin"] = df["label"].apply(lambda x: 1 if str(x).strip().lower() in ("spam", "1", "true") else 0)
    return df

# --- train_baseline.py logic encapsulated ---
def train_baseline_logic(data_url: str, out_dir: str):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.svm import LinearSVC
    from sklearn.linear_model import LogisticRegression

    df = load_data(data_url)
    X = df["message_clean"].values
    y = df["label_bin"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y if len(np.unique(y)) > 1 else None
    )

    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models_to_train = {"svm": LinearSVC(random_state=RANDOM_STATE), "logreg": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)}
    
    comparison = {"models": {}, "summary": {}}

    for model_type, model_instance in models_to_train.items():
        print(f"Training and evaluating model: {model_type}")
        model_instance.fit(X_train_tfidf, y_train)
        preds = model_instance.predict(X_test_tfidf)

        report = classification_report(y_test, preds, output_dict=True)
        cm = confusion_matrix(y_test, preds).tolist()

        metrics = {
            "classification_report": report,
            "confusion_matrix": cm,
            "model_type": model_type,
        }

        os.makedirs(out_dir, exist_ok=True)
        model_path = os.path.join(out_dir, f"model_{model_type}.joblib")
        vec_path = os.path.join(out_dir, f"vectorizer_{model_type}.joblib")
        metrics_path = os.path.join(out_dir, f"metrics_{model_type}.json")

        joblib.dump(model_instance, model_path)
        joblib.dump(vectorizer, vec_path)
        with open(metrics_path, "w", encoding="utf-8") as fh:
            json.dump(metrics, fh, indent=2)

        print("Training complete. Artifacts written to:")
        print(f"  model: {model_path}")
        print(f"  vectorizer: {vec_path}")
        print(f"  metrics: {metrics_path}")
        pprint(metrics)

        comparison["models"][model_type] = {
            "model_path": model_path,
            "vectorizer_path": vec_path,
            "metrics": metrics,
        }

    summary = {}
    for m, info in comparison["models"].items():
        metrics = info.get("metrics") or {}
        acc = None
        if isinstance(metrics.get("classification_report"), dict):
            acc = metrics["classification_report"].get("accuracy")
        summary[m] = {"accuracy": acc}

    comparison["summary"] = summary

    comp_path = os.path.join(out_dir, "comparison.json")
    with open(comp_path, "w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2)

    print(f"Wrote comparison report to: {comp_path}")
    return comp_path

# --- app.py logic encapsulated ---
def run_streamlit_app():
    st.title("Spam Classifier")
    st.write("Enter a message below to check if it's spam or not.")

    try:
        model_path = os.path.join("artifacts", "model_svm.joblib")
        vectorizer_path = os.path.join("artifacts", "vectorizer_svm.joblib") # Assuming SVM is the default model for the app

        model = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
    except FileNotFoundError:
        st.error("Model or vectorizer not found. Please ensure 'model_svm.joblib' and 'vectorizer_svm.joblib' are in the 'artifacts' directory.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model or vectorizer: {e}")
        st.stop()

    user_input = st.text_area("Message", "")

    if st.button("Classify"):
        if user_input:
            cleaned_input = clean_text(user_input)
            input_vectorized = vectorizer.transform([cleaned_input])
            prediction = model.predict(input_vectorized)
            if prediction[0] == 1:
                st.error("This message is SPAM!")
            else:
                st.success("This message is NOT SPAM.")
        else:
            st.warning("Please enter a message to classify.")

# --- diagnose_env.py logic encapsulated ---
def diagnose_env_logic():
    def try_import(module_name: str):
        t0 = time.time()
        try:
            mod = __import__(module_name)
            ok = True
            err = None
        except Exception as e:
            mod = None
            ok = False
            err = e
        t1 = time.time()
        return ok, mod, err, t1 - t0

    print("Python:", sys.version.replace('\\n', ' '))
    print("Platform:", platform.platform())
    print("Machine:", platform.machine())
    try:
        ver = subprocess.check_output(["cmd", "/c", "ver"], stderr=subprocess.STDOUT, encoding="utf-8")
        print("cmd /c ver ->", ver.strip())
    except Exception as e:
        print("Could not run 'cmd /c ver':", e)

    for m in ("numpy", "scipy", "sklearn", "pandas"):
        ok, mod, err, dt = try_import(m)
        if ok:
            try:
                ver = getattr(mod, "__version__", "<unknown>")
            except Exception:
                ver = "<unknown>"
            print(f"Imported {m} (v{ver}) in {dt:.3f}s")
        else:
            print(f"Failed to import {m} after {dt:.3f}s: {err!r}")
    print("Done diagnostics.")

# --- download_data.py logic encapsulated ---
def download_data_logic(url: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sms_spam_no_header.csv")
    try:
        df = pd.read_csv(url, header=None, names=["label", "message"], encoding="utf-8", engine="python")
    except Exception:
        df = pd.read_csv(url, header=None, names=["label", "message"], encoding="latin-1", engine="python")
    df.to_csv(out_path, index=False)
    print(f"Downloaded dataset to: {out_path}")
    return out_path

# --- predict_baseline.py logic encapsulated ---
def predict_baseline_logic(model_path: str, vectorizer_path: str, text: str = None, csv_path: str = None):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    if text:
        cleaned = clean_text(text)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        print(f"🔍 Single text prediction:\n  ➜ {text} → {'Spam' if pred == 1 else 'Ham'}")
    elif csv_path:
        df = pd.read_csv(csv_path)
        df["message_clean"] = df["message"].apply(clean_text)
        X_vec = vectorizer.transform(df["message_clean"])
        preds = model.predict(X_vec)
        df["predicted_label"] = preds
        out_path = csv_path.replace(".csv", "_predicted.csv")
        df.to_csv(out_path, index=False)
        print(f"✅ Predictions saved to: {out_path}")
    else:
        print("Please specify either --text or --csv.")

# --- Main dispatcher ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified script for spam classification tasks.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Streamlit app subcommand
    streamlit_parser = subparsers.add_parser("streamlit", help="Run the Streamlit spam classification app.")

    # Train subcommand
    train_parser = subparsers.add_parser("train", help="Train baseline models (SVM and LogReg) and generate a comparison report.")
    train_parser.add_argument("--data", default=DEFAULT_URL, help="Path or URL to dataset CSV (no header, label,message)")
    train_parser.add_argument("--out", default="artifacts", help="Output directory for artifacts")

    # Diagnose subcommand
    diagnose_parser = subparsers.add_parser("diagnose", help="Run environment diagnostics.")

    # Download subcommand
    download_parser = subparsers.add_parser("download", help="Download the dataset.")
    download_parser.add_argument("--url", default=DEFAULT_URL, help="Dataset URL")
    download_parser.add_argument("--out", default="data", help="Output directory")

    # Predict subcommand
    predict_parser = subparsers.add_parser("predict", help="Make predictions using a trained model.")
    predict_parser.add_argument("--model", required=True, help="Path to the trained model joblib file.")
    predict_parser.add_argument("--vectorizer", required=True, help="Path to the fitted vectorizer joblib file.")
    predict_parser.add_argument("--text", help="Single message to classify.")
    predict_parser.add_argument("--csv", help="Path to CSV file with column 'message' for batch prediction.")

    args = parser.parse_args()

    if args.command == "streamlit":
        run_streamlit_app()
    elif args.command == "train":
        train_baseline_logic(args.data, args.out)
    elif args.command == "diagnose":
        diagnose_env_logic()
    elif args.command == "download":
        download_data_logic(args.url, args.out)
    elif args.command == "predict":
        predict_baseline_logic(args.model, args.vectorizer, args.text, args.csv)
    else:
        parser.print_help()