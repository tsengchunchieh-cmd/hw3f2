Spam classification — Phase1 baseline

This change implements a reproducible baseline for spam classification using the public SMS spam dataset.

Quickstart (recommended: create a virtualenv first)

PowerShell example:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r openspec/changes/add-spam-classification/requirements.txt
python openspec/changes/add-spam-classification/scripts/download_data.py --out data
python openspec/changes/add-spam-classification/scripts/train_baseline.py --data data/sms_spam_no_header.csv --model svm --out artifacts
```

Outputs
- `artifacts/model_svm.joblib` — trained model (SVM)
- `artifacts/vectorizer.joblib` — TF-IDF vectorizer
- `artifacts/metrics.json` — evaluation metrics (classification report + confusion matrix)

Notes
- Default model is SVM; pass `--model logreg` to train logistic regression instead.
- The dataset is SMS spam (not email) — it still serves as a text spam classification baseline.
