"""Download the dataset to data/ directory.

Expected CSV (no header) with two columns: label,text
"""
import os
import argparse
import pandas as pd

DEFAULT_URL = (
    "https://raw.github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
)


def download(url: str, out_dir: str = "data") -> str:
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sms_spam_no_header.csv")
    # Read with pandas and write to ensure consistent formatting
    try:
        df = pd.read_csv(url, header=None, names=["label", "message"], encoding="utf-8", engine="python")
    except Exception:
        # fallback: try with latin-1
        df = pd.read_csv(url, header=None, names=["label", "message"], encoding="latin-1", engine="python")
    df.to_csv(out_path, index=False)
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL, help="Dataset URL")
    parser.add_argument("--out", default="data", help="Output directory")
    args = parser.parse_args()
    path = download(args.url, args.out)
    print(f"Downloaded dataset to: {path}")
