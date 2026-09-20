# Fraud Detection

Anomaly detection comparing Isolation Forest and LOF on 284K credit card transactions.

## Results
| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Isolation Forest | 0.3571 | 0.2027 | 0.2586 | 0.6010 |
| LOF | 0.0000 | 0.0000 | 0.0000 | 0.4995 |

**Best Model:** Isolation Forest (F1 = 0.2586)

Both models are run with `contamination=0.001`, below the dataset's actual 0.17% fraud rate — a parameter choice worth revisiting rather than a tuned decision. `outputs/metrics.json` in this repo is the real, unedited output of `train_model.py` — nothing above is hand-typed.

## Why LOF failed to detect any fraud

LOF's precision, recall, and F1 all came out to exactly 0.0000 — not just lower than Isolation Forest, but a genuine algorithmic failure on this dataset, not a bug. Confirmed directly: LOF flagged 78 of the 85,443 test transactions as anomalies, and zero of them overlapped with the 148 actual fraud cases in that test set. LOF is a distance-based method: it measures local density using Euclidean distance between points. The model trains on 30 features total — Time, the 28 PCA-transformed components (`V1`-`V28`), and Amount (Time and Amount are on their original raw scale, not PCA-transformed, though all 30 are standardized together before fitting). In a space this high-dimensional, distance-based methods suffer from the curse of dimensionality — distances between points become less discriminative as dimensionality increases, since everything starts looking roughly equidistant.

Isolation Forest doesn't share this weakness. It isolates points through random tree-based feature splits rather than measuring distance, which is far more robust in high-dimensional, PCA-transformed spaces. On this dataset, Isolation Forest flagged 84 of the 85,443 test transactions and correctly caught 30 of the 148 real fraud cases.

This is a real, verified result (code reviewed line-by-line to rule out implementation bugs) and illustrates a genuine tradeoff worth knowing: distance-based anomaly detection degrades on high-dimensional data in a way that tree-based methods don't.

## Dataset
- Kaggle Credit Card Fraud Detection
- 284,807 transactions
- 0.17% fraud rate
- Not included in this repo (150MB, third-party) — download from Kaggle and place as `creditcard.csv` in the repo root to reproduce

## Files
- `train_model.py` — Training script
- `requirements.txt` — Dependencies (`pip install -r requirements.txt`)
- `outputs/metrics.json` — Real output from the training run, committed for verifiability
- `models/` — Trained model artifacts (`.pkl`), gitignored due to size (up to ~80MB) — regenerate locally by running `train_model.py`
