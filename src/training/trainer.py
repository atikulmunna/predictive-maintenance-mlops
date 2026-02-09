"""Unified training entrypoint for model pipelines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.training.ensemble_pipeline import run_ensemble_pipeline
from src.training.lstm_pipeline import run_lstm_pipeline
from src.training.mlflow_utils import write_registry_state
from src.training.xgboost_pipeline import run_xgboost_pipeline


def run_selected_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    model = args.model.lower()
    data_dir = Path(args.data_dir)

    if model == "xgboost":
        return {
            "xgboost": run_xgboost_pipeline(
                data_dir=data_dir,
                enable_mlflow=not args.no_mlflow,
                register_model=args.register_models,
                registered_model_name=args.xgb_registered_model_name,
            )
        }

    if model == "lstm":
        return {
            "lstm": run_lstm_pipeline(
                data_dir=data_dir,
                sequence_length=args.sequence_length,
                top_k_features=args.top_k_features,
                epochs=args.epochs,
                batch_size=args.batch_size,
                enable_mlflow=not args.no_mlflow,
                register_model=args.register_models,
                registered_model_name=args.lstm_registered_model_name,
            )
        }

    if model == "ensemble":
        payload = run_ensemble_pipeline(
            data_dir=data_dir,
            min_f2_gain=args.min_f2_gain,
            allow_calibration=not args.no_calibration,
            enable_mlflow=not args.no_mlflow,
        )
        (data_dir / "models").mkdir(parents=True, exist_ok=True)
        (data_dir / "models" / "ensemble_metrics.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        return {"ensemble": payload}

    if model == "all":
        results: dict[str, Any] = {}
        results["xgboost"] = run_xgboost_pipeline(
            data_dir=data_dir,
            enable_mlflow=not args.no_mlflow,
            register_model=args.register_models,
            registered_model_name=args.xgb_registered_model_name,
        )
        results["lstm"] = run_lstm_pipeline(
            data_dir=data_dir,
            sequence_length=args.sequence_length,
            top_k_features=args.top_k_features,
            epochs=args.epochs,
            batch_size=args.batch_size,
            enable_mlflow=not args.no_mlflow,
            register_model=args.register_models,
            registered_model_name=args.lstm_registered_model_name,
        )
        results["ensemble"] = run_ensemble_pipeline(
            data_dir=data_dir,
            min_f2_gain=args.min_f2_gain,
            allow_calibration=not args.no_calibration,
            enable_mlflow=not args.no_mlflow,
        )
        (data_dir / "models").mkdir(parents=True, exist_ok=True)
        (data_dir / "models" / "ensemble_metrics.json").write_text(
            json.dumps(results["ensemble"], indent=2), encoding="utf-8"
        )
        _write_local_registry_state(data_dir=data_dir, results=results)
        return results

    raise ValueError(f"Unsupported model: {args.model}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run training pipelines for predictive maintenance.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["xgboost", "lstm", "ensemble", "all"],
        help="Pipeline to run",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Path to data directory")

    # LSTM options
    parser.add_argument("--epochs", type=int, default=100, help="LSTM max epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="LSTM batch size")
    parser.add_argument("--sequence-length", type=int, default=30, help="LSTM sequence length")
    parser.add_argument("--top-k-features", type=int, default=40, help="Top correlated features for LSTM")

    # Ensemble options
    parser.add_argument("--min-f2-gain", type=float, default=0.005, help="Min val F2 gain to select ensemble")
    parser.add_argument("--no-calibration", action="store_true", help="Disable Platt calibration")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    parser.add_argument("--register-models", action="store_true", help="Register XGBoost/LSTM models in MLflow")
    parser.add_argument(
        "--xgb-registered-model-name",
        default="predictive-maintenance-xgboost",
        help="MLflow registered model name for XGBoost",
    )
    parser.add_argument(
        "--lstm-registered-model-name",
        default="predictive-maintenance-lstm",
        help="MLflow registered model name for LSTM",
    )
    return parser


def _write_local_registry_state(data_dir: Path, results: dict[str, Any]) -> None:
    xgb = results.get("xgboost", {})
    lstm = results.get("lstm", {})
    ens = results.get("ensemble", {})
    payload = {
        "selected_model": ens.get("selected_model"),
        "selection_threshold": ens.get("selected_threshold"),
        "selected_test_f2": ens.get("test_f2"),
        "selection_source": "ensemble_metrics",
        "xgboost": {
            "test_f2": xgb.get("test_f2"),
            "mlflow_run_id": xgb.get("mlflow_run_id"),
            "mlflow_registry": xgb.get("mlflow_registry"),
        },
        "lstm": {
            "test_f2": lstm.get("test_f2"),
            "mlflow_run_id": lstm.get("mlflow_run_id"),
            "mlflow_registry": lstm.get("mlflow_registry"),
        },
        "ensemble": {
            "test_f2": ens.get("test_f2"),
            "mlflow_run_id": ens.get("mlflow_run_id"),
        },
    }
    write_registry_state(data_dir=data_dir, payload=payload)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    results = run_selected_pipeline(args)

    print("Pipeline run complete.")
    for name, payload in results.items():
        if name == "ensemble":
            print(f"- {name}: selected_model={payload['selected_model']}, test_f2={payload['test_f2']:.4f}")
        else:
            print(f"- {name}: test_f2={payload['test_f2']:.4f}")


if __name__ == "__main__":
    main()
