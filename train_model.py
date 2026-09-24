"""
Fraud Detection Model: Isolation Forest and LOF

Trains and compares two unsupervised anomaly detection models on the
Kaggle Credit Card Fraud Detection dataset, then saves the trained
models and evaluation metrics.
"""

import json
import os
import pickle
from typing import Dict

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

DATA_PATH = "creditcard.csv"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

RANDOM_STATE = 42
TEST_SIZE = 0.3
CONTAMINATION = 0.001
LOF_N_NEIGHBORS = 20


def load_data(path: str) -> pd.DataFrame:
    """Load the raw transactions dataset."""
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    print(f"Data loaded. Shape: {df.shape}")
    return df


def prepare_data(df: pd.DataFrame):
    """Split features and target, scale features, and train/test split."""
    print("Preparing data...")
    X = df.drop("Class", axis=1)
    y = df["Class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print("Data prepared.")
    return X_train, X_test, y_train, y_test, scaler


def train_isolation_forest(X_train):
    print("Training Isolation Forest...")
    model = IsolationForest(
        contamination=CONTAMINATION, random_state=RANDOM_STATE, n_jobs=-1, verbose=0
    )
    model.fit(X_train)
    return model


def train_lof(X_train):
    print("Training Local Outlier Factor (LOF)...")
    model = LocalOutlierFactor(
        n_neighbors=LOF_N_NEIGHBORS, contamination=CONTAMINATION, novelty=True, n_jobs=-1
    )
    model.fit(X_train)
    return model


def get_metrics(y_true, y_pred) -> Dict[str, float]:
    """Compute precision, recall, F1, and ROC AUC for a set of predictions."""
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred),
    }


def print_comparison(metrics_iso: Dict[str, float], metrics_lof: Dict[str, float]) -> None:
    print("\nResults, model comparison")
    print(f"{'Model':<20} {'Precision':<15} {'Recall':<15} {'F1':<15} {'ROC_AUC':<15}")
    print(
        f"{'Isolation Forest':<20} {metrics_iso['precision']:<15.4f} "
        f"{metrics_iso['recall']:<15.4f} {metrics_iso['f1']:<15.4f} {metrics_iso['roc_auc']:<15.4f}"
    )
    print(
        f"{'LOF':<20} {metrics_lof['precision']:<15.4f} {metrics_lof['recall']:<15.4f} "
        f"{metrics_lof['f1']:<15.4f} {metrics_lof['roc_auc']:<15.4f}"
    )


def save_models(iso_forest, lof, scaler, models_dir: str) -> None:
    os.makedirs(models_dir, exist_ok=True)
    with open(os.path.join(models_dir, "isolation_forest.pkl"), "wb") as f:
        pickle.dump(iso_forest, f)
    print("Isolation Forest saved")

    with open(os.path.join(models_dir, "lof_model.pkl"), "wb") as f:
        pickle.dump(lof, f)
    print("LOF saved")

    with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    print("Scaler saved")


def save_metrics(
    best_model: str,
    metrics_iso: Dict[str, float],
    metrics_lof: Dict[str, float],
    df: pd.DataFrame,
    outputs_dir: str,
) -> None:
    os.makedirs(outputs_dir, exist_ok=True)
    metrics_all = {
        "best_model": best_model,
        "models": {"isolation_forest": metrics_iso, "lof": metrics_lof},
        "dataset": {
            "total_transactions": len(df),
            "fraud_cases": int(df["Class"].sum()),
            "fraud_rate": f"{(df['Class'].sum() / len(df)) * 100:.2f}%",
        },
    }
    with open(os.path.join(outputs_dir, "metrics.json"), "w") as f:
        json.dump(metrics_all, f, indent=2)
    print("Metrics saved")


def main() -> None:
    print("Fraud detection model training: 2 algorithms")

    df = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test, scaler = prepare_data(df)

    iso_forest = train_isolation_forest(X_train)
    y_pred_iso = (iso_forest.predict(X_test) == -1).astype(int)
    print(f"Isolation Forest trained. Anomalies flagged: {y_pred_iso.sum()}")

    lof = train_lof(X_train)
    y_pred_lof = (lof.predict(X_test) == -1).astype(int)
    print(f"LOF trained. Anomalies flagged: {y_pred_lof.sum()}")

    print("Calculating metrics...")
    metrics_iso = get_metrics(y_test, y_pred_iso)
    metrics_lof = get_metrics(y_test, y_pred_lof)
    print_comparison(metrics_iso, metrics_lof)

    best_model = "Isolation Forest" if metrics_iso["f1"] > metrics_lof["f1"] else "LOF"
    print(f"\nBest model: {best_model}")

    print("Saving models...")
    save_models(iso_forest, lof, scaler, MODELS_DIR)
    save_metrics(best_model, metrics_iso, metrics_lof, df, OUTPUTS_DIR)

    print("Training complete.")
    print(
        "Files created: models/isolation_forest.pkl, models/lof_model.pkl, "
        "models/scaler.pkl, outputs/metrics.json"
    )
    print(f"Best model: {best_model}")


if __name__ == "__main__":
    main()
