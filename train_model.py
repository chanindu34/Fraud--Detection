"""
Fraud Detection Model - Isolation Forest & LOF
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import pickle
import json
import os

print("="*70)
print("FRAUD DETECTION MODEL TRAINING - 2 ALGORITHMS")
print("="*70)

os.makedirs('outputs', exist_ok=True)
os.makedirs('models', exist_ok=True)

# Load data
print("\n📥 Loading data...")
df = pd.read_csv('creditcard.csv')
print(f"✅ Data loaded! Shape: {df.shape}")

# Prepare data
print("\n🔧 Preparing data...")
X = df.drop('Class', axis=1)
y = df['Class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)
print(f"✅ Data prepared!")

# Model 1: Isolation Forest
print("\n" + "="*70)
print("MODEL 1: ISOLATION FOREST")
print("="*70)
iso_forest = IsolationForest(contamination=0.001, random_state=42, n_jobs=-1, verbose=0)
iso_forest.fit(X_train)
y_pred_iso = (iso_forest.predict(X_test) == -1).astype(int)
print(f"✅ Isolation Forest trained! Anomalies: {y_pred_iso.sum()}")

# Model 2: LOF
print("\n" + "="*70)
print("MODEL 2: LOCAL OUTLIER FACTOR (LOF)")
print("="*70)
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.001, novelty=True, n_jobs=-1)
lof.fit(X_train)
y_pred_lof = (lof.predict(X_test) == -1).astype(int)
print(f"✅ LOF trained! Anomalies: {y_pred_lof.sum()}")

# Calculate metrics
print("\n🔮 Calculating metrics...")
def get_metrics(y_true, y_pred):
    return {
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred)
    }

metrics_iso = get_metrics(y_test, y_pred_iso)
metrics_lof = get_metrics(y_test, y_pred_lof)

# Display results
print("\n" + "="*80)
print("RESULTS - MODEL COMPARISON")
print("="*80)
print(f"\n{'Model':<20} {'Precision':<15} {'Recall':<15} {'F1':<15} {'ROC-AUC':<15}")
print("-"*80)
print(f"{'Isolation Forest':<20} {metrics_iso['precision']:<15.4f} {metrics_iso['recall']:<15.4f} {metrics_iso['f1']:<15.4f} {metrics_iso['roc_auc']:<15.4f}")
print(f"{'LOF':<20} {metrics_lof['precision']:<15.4f} {metrics_lof['recall']:<15.4f} {metrics_lof['f1']:<15.4f} {metrics_lof['roc_auc']:<15.4f}")

best_model = 'Isolation Forest' if metrics_iso['f1'] > metrics_lof['f1'] else 'LOF'
print(f"\n🏆 BEST MODEL: {best_model}")

# Save models
print("\n" + "="*70)
print("SAVING MODELS")
print("="*70)

with open('models/isolation_forest.pkl', 'wb') as f:
    pickle.dump(iso_forest, f)
print("✅ Isolation Forest saved")

with open('models/lof_model.pkl', 'wb') as f:
    pickle.dump(lof, f)
print("✅ LOF saved")

with open('models/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Scaler saved")

# Save metrics
metrics_all = {
    'best_model': best_model,
    'models': {
        'isolation_forest': metrics_iso,
        'lof': metrics_lof
    },
    'dataset': {
        'total_transactions': len(df),
        'fraud_cases': int(df['Class'].sum()),
        'fraud_rate': f"{(df['Class'].sum() / len(df)) * 100:.2f}%"
    }
}

with open('outputs/metrics.json', 'w') as f:
    json.dump(metrics_all, f, indent=2)
print("✅ Metrics saved")

print("\n" + "="*70)
print("TRAINING COMPLETE! ✅")
print("="*70)
print(f"\n📁 FILES CREATED:")
print(f"   ✅ models/isolation_forest.pkl")
print(f"   ✅ models/lof_model.pkl")
print(f"   ✅ models/scaler.pkl")
print(f"   ✅ outputs/metrics.json")
print(f"\n🏆 BEST MODEL: {best_model}")
print("\n✅ DONE!")