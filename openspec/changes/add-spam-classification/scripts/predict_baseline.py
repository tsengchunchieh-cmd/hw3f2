import os
import argparse
import pandas as pd
import joblib

from train_baseline import clean_text

def predict_text(model_path, vectorizer_path, text=None, csv_path=None):
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--vectorizer", required=True)
    parser.add_argument("--text", help="Single message to classify")
    parser.add_argument("--csv", help="Path to CSV file with column 'message'")
    args = parser.parse_args()
    predict_text(args.model, args.vectorizer, text=args.text, csv_path=args.csv)
