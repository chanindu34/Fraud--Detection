# Fraud Detection

Anomaly detection comparing Isolation Forest and LOF on 284K credit card transactions.

## Results
| Model | Precision | Recall | F1-Score | ROC-AUC |
|-------|-----------|--------|----------|---------|
| Isolation Forest | 0.3571 | 0.2027 | 0.2586 | 0.6010 |
| LOF | 0.0000 | 0.0000 | 0.0000 | 0.4995 |

**Best Model:** Isolation Forest (F1 = 0.2586)

## Dataset
- Kaggle Credit Card Fraud Detection
- 284,807 transactions
- 0.17% fraud rate

## Files
- `train_model.py` — Training script
- `models/` — Trained models
- `outputs/metrics.json` — Results