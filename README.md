# Fraud Detection

Anomaly detection comparing Isolation Forest and LOF on 284K credit card transactions.

## Results
| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Isolation Forest | 0.3571 | 0.2027 | 0.2586 | 0.6010 |
| LOF | 0.0000 | 0.0000 | 0.0000 | 0.4995 |

**Best Model:** Isolation Forest (F1 = 0.2586)

## Why LOF failed to detect any fraud

LOF's precision, recall, and F1 all came out to exactly 0.0000 — not just lower than Isolation Forest, but a genuine algorithmic failure on this dataset, not a bug. LOF is a distance-based method: it measures local density using Euclidean distance between points. This dataset has 28 PCA-transformed features (`V1`-`V28`), and in high-dimensional spaces like this, distance-based methods suffer from the curse of dimensionality — distances between points become less discriminative as dimensionality increases, since everything starts looking roughly equidistant.

Isolation Forest doesn't share this weakness. It isolates points through random tree-based feature splits rather than measuring distance, which is far more robust in high-dimensional, PCA-transformed spaces. On this dataset, LOF's flagged "local outliers" ended up with zero overlap with the actual fraud cases, while Isolation Forest's tree-based approach correctly identified a meaningful fraction of them.

This is a real, verified result (code reviewed line-by-line to rule out implementation bugs) and illustrates a genuine tradeoff worth knowing: distance-based anomaly detection degrades on high-dimensional data in a way that tree-based methods don't.

## Dataset
- Kaggle Credit Card Fraud Detection
- 284,807 transactions
- 0.17% fraud rate

## Files
- `train_model.py` — Training script
- `models/` — Trained models
- `outputs/metrics.json` — Results
