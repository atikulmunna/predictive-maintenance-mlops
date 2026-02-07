"""Train/evaluate/save the XGBoost baseline model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


DEFAULT_PARAMS = {
    "n_estimators": 248,
    "max_depth": 8,
    "learning_rate": 0.05841217467897381,
    "subsample": 0.9827401966626917,
    "colsample_bytree": 0.9025335089530546,
    "min_child_weight": 2,
    "gamma": 0.9985060446560522,
    "reg_alpha": 0.3728883454481621,
    "reg_lambda": 0.1348470231205584,
    "random_state": 42,
    "tree_method": "hist",
    "eval_metric": "logloss",
}


def load_training_params(models_dir: Path) -> dict[str, Any]:
    metrics_path = models_dir / "xgboost_baseline_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        params = payload.get("hyperparameters")
        if isinstance(params, dict) and params:
            return params
    return DEFAULT_PARAMS.copy()


def split_by_engine(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    engines = df["unit_id"].unique()
    n_engines = len(engines)
    train_engines = engines[: int(0.70 * n_engines)]
    val_engines = engines[int(0.70 * n_engines) : int(0.85 * n_engines)]
    test_engines = engines[int(0.85 * n_engines) :]
    return (
        df["unit_id"].isin(train_engines),
        df["unit_id"].isin(val_engines),
        df["unit_id"].isin(test_engines),
    )


def run_xgboost_pipeline(data_dir: Path) -> dict[str, Any]:
    processed_path = data_dir / "processed" / "train_features_FD001.csv"
    models_dir = data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_path)
    exclude_cols = ["unit_id", "cycle", "RUL", "failure_soon"]
    feature_cols = [col for col in df.columns if col not in exclude_cols]

    x = df[feature_cols]
    y = df["failure_soon"]

    train_mask, val_mask, test_mask = split_by_engine(df)
    x_train, y_train = x[train_mask], y[train_mask]
    x_val, y_val = x[val_mask], y[val_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    smote = SMOTE(random_state=42, k_neighbors=5)
    x_train_balanced, y_train_balanced = smote.fit_resample(x_train_scaled, y_train)

    params = load_training_params(models_dir)
    params.update({"random_state": 42, "tree_method": "hist", "eval_metric": "logloss"})

    model = xgb.XGBClassifier(**params)
    model.fit(
        x_train_balanced,
        y_train_balanced,
        eval_set=[(x_val_scaled, y_val)],
        verbose=False,
    )

    y_val_pred = model.predict(x_val_scaled)
    y_val_proba = model.predict_proba(x_val_scaled)[:, 1]
    y_test_pred = model.predict(x_test_scaled)
    y_test_proba = model.predict_proba(x_test_scaled)[:, 1]

    f2_val = float(fbeta_score(y_val, y_val_pred, beta=2))
    precision_val = float(precision_score(y_val, y_val_pred))
    recall_val = float(recall_score(y_val, y_val_pred))
    roc_auc_val = float(roc_auc_score(y_val, y_val_proba))

    f2_test = float(fbeta_score(y_test, y_test_pred, beta=2))
    precision_test = float(precision_score(y_test, y_test_pred))
    recall_test = float(recall_score(y_test, y_test_pred))
    roc_auc_test = float(roc_auc_score(y_test, y_test_proba))

    model_path = models_dir / "xgboost_baseline.json"
    scaler_path = models_dir / "scaler.pkl"
    feature_path = models_dir / "feature_names.json"
    metrics_path = models_dir / "xgboost_baseline_metrics.json"

    model.save_model(model_path)
    joblib.dump(scaler, scaler_path)
    with open(feature_path, "w", encoding="utf-8") as f:
        json.dump({"features": feature_cols}, f, indent=2)

    metrics = {
        "model": "XGBoost Baseline",
        "timestamp": pd.Timestamp.now().isoformat(),
        "validation_metrics": {
            "f2_score": f2_val,
            "precision": precision_val,
            "recall": recall_val,
            "roc_auc": roc_auc_val,
        },
        "test_metrics": {
            "f2_score": f2_test,
            "precision": precision_test,
            "recall": recall_test,
            "roc_auc": roc_auc_test,
        },
        "hyperparameters": params,
        "dataset_info": {
            "train_samples": int(len(x_train_balanced)),
            "val_samples": int(len(x_val)),
            "test_samples": int(len(x_test)),
            "num_features": int(len(feature_cols)),
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "feature_path": str(feature_path),
        "metrics_path": str(metrics_path),
        "validation_f2": f2_val,
        "test_f2": f2_test,
        "test_precision": precision_test,
        "test_recall": recall_test,
        "test_roc_auc": roc_auc_test,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and persist XGBoost baseline artifacts.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Path to data directory")
    args = parser.parse_args()

    result = run_xgboost_pipeline(args.data_dir)
    print(f"Saved model: {result['model_path']}")
    print(f"Saved scaler: {result['scaler_path']}")
    print(f"Saved feature names: {result['feature_path']}")
    print(f"Saved metrics: {result['metrics_path']}")
    print(
        "Test metrics: "
        f"F2={result['test_f2']:.4f}, "
        f"Precision={result['test_precision']:.4f}, "
        f"Recall={result['test_recall']:.4f}, "
        f"ROC-AUC={result['test_roc_auc']:.4f}"
    )


if __name__ == "__main__":
    main()

