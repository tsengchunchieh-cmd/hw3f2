## Why
The project owner wants to add a machine-learning capability for spam email classification. This capability will allow experimenting with simple ML models and creating a baseline that can be iterated on in later phases.

Primary motivation:
- Establish a reproducible ML baseline for spam detection using the provided public dataset.
- Provide a clear spec-driven change so implementations and evaluations are tracked and reviewed.

## What Changes
- Add a new capability `spam` with a baseline implementation in Phase 1.
- Phase 1 (baseline): Download dataset from the provided URL, preprocess, and build a basic spam classifier using SVM as the initial baseline. (The project goal also includes exploring logistic regression; this will be part of later phases or comparisons.)
- Add specs for the new capability and an implementation tasks checklist.

## Data Source
Dataset (CSV):
https://raw.github.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv

## Phases
- Phase1-baseline: Basic spam classifier (SVM baseline using the dataset above). Deliverables: preprocessing script, training script/notebook, evaluation metrics, saved model artifact, README describing how to reproduce.
- Phase2: (empty placeholder)
- Phase3: (empty placeholder)

## Impact
- Affected specs: adds a new capability `spam` under `openspec/changes/add-spam-classification/specs/spam/spec.md` as an ADDED requirement.
- Affected code: new training and evaluation scripts, optional notebook, and a `requirements.txt` or `package.json` describing dependencies.
- Human reviewers: ML reviewer and documentation reviewer should confirm preprocessing and evaluation choices.

## Rollout
1. Review and refine the proposal and spec deltas.
2. Approve the proposal.
3. Implement Phase1 tasks from `tasks.md` and attach results (notebook, metrics, artifacts).
4. Validate and, if successful, archive the change following OpenSpec rules.

## Risks / Notes
- The dataset appears to be SMS spam (not email) — confirm if the dataset is acceptable for 'email' classification experiments or if the user intended SMS spam. The dataset will still serve as a spam classification baseline for text classification.
- Small datasets can overfit; keep evaluation strict (train/test split, cross-validation).

***
Created as an OpenSpec proposal by agent on user request. Please review and confirm whether you want SVM as the baseline or logistic regression for Phase1.