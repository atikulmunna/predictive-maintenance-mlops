"""One-command end-to-end validation against real project artifacts.

Checks:
1. Artifact presence under data/models
2. API /health, /predict, /explain (via FastAPI TestClient)
3. Drift detection report generation
4. Conditional retraining decision generation

Optional:
- Run short local training orchestration before checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi.testclient import TestClient

# Ensure local `src` package is importable when script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.monitoring.drift_detection import run_drift_detection
from src.pipelines.prefect_flow import training_flow_local
from src.pipelines.retraining_pipeline import run_retraining_decision
from src.serving.api import create_app


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


def _load_feature_names(models_dir: Path) -> list[str]:
    payload = json.loads((models_dir / "feature_names.json").read_text(encoding="utf-8"))
    return [str(x) for x in payload["features"]]


def _load_sequence_length(models_dir: Path) -> int:
    payload = json.loads((models_dir / "lstm_features.json").read_text(encoding="utf-8"))
    return int(payload.get("sequence_length", 30))


def _build_sequence(data_path: Path, features: list[str], seq_len: int) -> list[dict[str, float]]:
    df = pd.read_csv(data_path)
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Input data missing required feature columns. Example: {missing_cols[:5]}")

    window = df[features].head(seq_len).copy()
    if window.empty:
        raise ValueError(f"No rows found in {data_path}")
    while len(window) < seq_len:
        window.loc[len(window)] = window.iloc[-1]

    return [
        {k: float(v) for k, v in row.items()}
        for row in window.to_dict(orient="records")
    ]


def run_e2e_check(
    *,
    data_dir: Path = Path("data"),
    features_csv: Path = Path("data/processed/train_features_FD001.csv"),
    run_training: bool = False,
    epochs: int = 2,
    batch_size: int = 64,
) -> dict[str, Any]:
    models_dir = data_dir / "models"

    required = [
        models_dir / "feature_names.json",
        models_dir / "lstm_features.json",
        models_dir / "ensemble_metrics.json",
        models_dir / "xgboost_baseline.json",
        models_dir / "scaler.pkl",
        models_dir / "lstm_temporal.h5",
        models_dir / "lstm_scaler.pkl",
    ]
    for p in required:
        _require(p)
    _require(features_csv)

    if run_training:
        training_flow_local(
            data_dir=str(data_dir),
            epochs=epochs,
            batch_size=batch_size,
            no_mlflow=True,
        )

    features = _load_feature_names(models_dir)
    seq_len = _load_sequence_length(models_dir)
    sequence = _build_sequence(features_csv, features, seq_len)

    app = create_app(data_dir=data_dir)
    with TestClient(app) as client:
        health = client.get("/health")
        if health.status_code != 200:
            raise RuntimeError(f"/health failed: {health.status_code} {health.text}")
        health_body = health.json()

        predict_payload = {"equipment_id": "e2e_real_engine_001", "sequence": sequence}
        predict = client.post("/predict", json=predict_payload)
        if predict.status_code != 200:
            raise RuntimeError(f"/predict failed: {predict.status_code} {predict.text}")
        predict_body = predict.json()

        explain = client.post("/explain?top_k=5", json=predict_payload)
        if explain.status_code != 200:
            raise RuntimeError(f"/explain failed: {explain.status_code} {explain.text}")
        explain_body = explain.json()

    drift = run_drift_detection(
        reference_path=features_csv,
        current_path=features_csv,
        data_dir=data_dir,
    )
    retrain = run_retraining_decision(
        data_dir=data_dir,
        engine="local",
        no_mlflow=True,
    )

    summary = {
        "health": {
            "status": health_body.get("status"),
            "selected_model_policy": health_body.get("selected_model_policy"),
            "policy_source": health_body.get("policy_source"),
        },
        "predict": {
            "model_used": predict_body.get("model_used"),
            "failure_probability": predict_body.get("failure_probability"),
            "failure_prediction": predict_body.get("failure_prediction"),
        },
        "explain": {
            "top_k": explain_body.get("top_k"),
            "contribution_count": len(explain_body.get("contributions", [])),
        },
        "drift": drift.get("summary", {}),
        "retraining": {
            "triggered": retrain.get("retraining_triggered"),
            "reason": retrain.get("reason"),
        },
    }

    out_path = models_dir / "e2e_real_check_report.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report_path"] = str(out_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one-command E2E checks on real project data/artifacts.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Project data directory")
    parser.add_argument(
        "--features-csv",
        type=Path,
        default=Path("data/processed/train_features_FD001.csv"),
        help="CSV used to build a real sequence payload",
    )
    parser.add_argument("--run-training", action="store_true", help="Run short local training before checks")
    parser.add_argument("--epochs", type=int, default=2, help="LSTM epochs when --run-training is set")
    parser.add_argument("--batch-size", type=int, default=64, help="LSTM batch size when --run-training is set")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_e2e_check(
        data_dir=args.data_dir,
        features_csv=args.features_csv,
        run_training=args.run_training,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print("E2E real-data check complete.")
    print(f"- health status: {payload['health']['status']}")
    print(f"- predict model_used: {payload['predict']['model_used']}")
    print(f"- explain contributions: {payload['explain']['contribution_count']}")
    print(f"- drift status: {payload['drift']['status']}")
    print(f"- retraining triggered: {payload['retraining']['triggered']} ({payload['retraining']['reason']})")
    print(f"- report: {payload['report_path']}")


if __name__ == "__main__":
    main()
