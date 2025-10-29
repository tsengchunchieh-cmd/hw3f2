## ADDED Requirements

### Requirement: Spam Classification Capability (Baseline)
The system SHALL provide a reproducible baseline text classification pipeline that can classify messages as spam or ham using a provided labeled dataset.

#### Scenario: Dataset ingest and baseline training
- **WHEN** the dataset CSV at the provided URL is downloaded and preprocessed (cleaned, tokenized, vectorized)
- **THEN** the baseline training script SHALL train a classifier (SVM baseline) and produce a saved model artifact and evaluation metrics (precision, recall, F1)

#### Scenario: Evaluation artifact produced
- **WHEN** training completes
- **THEN** the system SHALL save evaluation metrics to `artifacts/metrics.json` and persist the model and vectorizer into `artifacts/` for reproducibility

### Requirement: Reproducibility
The system SHALL record random seed and package environment (e.g., `requirements.txt` or `pyproject.toml`) so the baseline can be reproduced.

#### Scenario: Reproducible run
- **WHEN** a user follows the README reproduction steps
- **THEN** they SHALL be able to reproduce training and evaluation results within reasonable variance
