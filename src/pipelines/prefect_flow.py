"""Phase 3 orchestration entrypoint for training pipelines.

Supports:
- local execution (no Prefect dependency required)
- Prefect flow execution (when Prefect is installed)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from src.training.ensemble_pipeline import run_ensemble_pipeline
from src.training.lstm_pipeline import run_lstm_pipeline
from src.training.xgboost_pipeline import run_xgboost_pipeline

try:
    from prefect import flow, get_run_logger, task

    PREFECT_AVAILABLE = True
except Exception:
    PREFECT_AVAILABLE = False
    flow = None
    task = None
    get_run_logger = None


def _logger() -> logging.Logger:
    if PREFECT_AVAILABLE and get_run_logger is not None:
        return get_run_logger()
    return logging.getLogger("pipeline_orchestrator")


def _run_xgboost_step(
    data_dir: Path,
    enable_mlflow: bool,
    register_model: bool,
    registered_model_name: str,
) -> dict[str, Any]:
    return run_xgboost_pipeline(
        data_dir=data_dir,
        enable_mlflow=enable_mlflow,
        register_model=register_model,
        registered_model_name=registered_model_name,
    )


def _run_lstm_step(
    data_dir: Path,
    sequence_length: int,
    top_k_features: int,
    epochs: int,
    batch_size: int,
    enable_mlflow: bool,
    register_model: bool,
    registered_model_name: str,
) -> dict[str, Any]:
    return run_lstm_pipeline(
        data_dir=data_dir,
        sequence_length=sequence_length,
        top_k_features=top_k_features,
        epochs=epochs,
        batch_size=batch_size,
        enable_mlflow=enable_mlflow,
        register_model=register_model,
        registered_model_name=registered_model_name,
    )


def _run_ensemble_step(
    data_dir: Path,
    min_f2_gain: float,
    allow_calibration: bool,
    enable_mlflow: bool,
) -> dict[str, Any]:
    payload = run_ensemble_pipeline(
        data_dir=data_dir,
        min_f2_gain=min_f2_gain,
        allow_calibration=allow_calibration,
        enable_mlflow=enable_mlflow,
    )
    (data_dir / "models").mkdir(parents=True, exist_ok=True)
    (data_dir / "models" / "ensemble_metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


if PREFECT_AVAILABLE:

    @task(name="train_xgboost", retries=1, retry_delay_seconds=10)
    def train_xgboost_task(data_dir: Path, enable_mlflow: bool) -> dict[str, Any]:
        return _run_xgboost_step(
            data_dir=data_dir,
            enable_mlflow=enable_mlflow,
            register_model=False,
            registered_model_name="predictive-maintenance-xgboost",
        )

    @task(name="train_lstm", retries=1, retry_delay_seconds=10)
    def train_lstm_task(
        data_dir: Path,
        sequence_length: int,
        top_k_features: int,
        epochs: int,
        batch_size: int,
        enable_mlflow: bool,
    ) -> dict[str, Any]:
        return _run_lstm_step(
            data_dir=data_dir,
            sequence_length=sequence_length,
            top_k_features=top_k_features,
            epochs=epochs,
            batch_size=batch_size,
            enable_mlflow=enable_mlflow,
            register_model=False,
            registered_model_name="predictive-maintenance-lstm",
        )

    @task(name="train_ensemble", retries=1, retry_delay_seconds=10)
    def train_ensemble_task(
        data_dir: Path,
        min_f2_gain: float,
        allow_calibration: bool,
        enable_mlflow: bool,
    ) -> dict[str, Any]:
        return _run_ensemble_step(
            data_dir=data_dir,
            min_f2_gain=min_f2_gain,
            allow_calibration=allow_calibration,
            enable_mlflow=enable_mlflow,
        )

    @flow(name="predictive-maintenance-training-flow")
    def training_flow_prefect(
        data_dir: str = "data",
        sequence_length: int = 30,
        top_k_features: int = 40,
        epochs: int = 100,
        batch_size: int = 32,
        min_f2_gain: float = 0.005,
        no_calibration: bool = False,
        no_mlflow: bool = False,
        register_models: bool = False,
        xgb_registered_model_name: str = "predictive-maintenance-xgboost",
        lstm_registered_model_name: str = "predictive-maintenance-lstm",
    ) -> dict[str, Any]:
        data_path = Path(data_dir)
        enable_mlflow = not no_mlflow
        allow_calibration = not no_calibration

        log = _logger()
        log.info("Starting Prefect training flow")
        xgb = _run_xgboost_step(
            data_dir=data_path,
            enable_mlflow=enable_mlflow,
            register_model=register_models,
            registered_model_name=xgb_registered_model_name,
        )
        lstm = _run_lstm_step(
            data_dir=data_path,
            sequence_length=sequence_length,
            top_k_features=top_k_features,
            epochs=epochs,
            batch_size=batch_size,
            enable_mlflow=enable_mlflow,
            register_model=register_models,
            registered_model_name=lstm_registered_model_name,
        )
        ensemble = train_ensemble_task(data_path, min_f2_gain, allow_calibration, enable_mlflow)
        log.info(
            "Flow complete | xgb_f2=%.4f lstm_f2=%.4f selected=%s selected_f2=%.4f",
            xgb["test_f2"],
            lstm["test_f2"],
            ensemble["selected_model"],
            ensemble["test_f2"],
        )
        return {"xgboost": xgb, "lstm": lstm, "ensemble": ensemble}


def training_flow_local(
    data_dir: str = "data",
    sequence_length: int = 30,
    top_k_features: int = 40,
    epochs: int = 100,
    batch_size: int = 32,
    min_f2_gain: float = 0.005,
    no_calibration: bool = False,
    no_mlflow: bool = False,
    register_models: bool = False,
    xgb_registered_model_name: str = "predictive-maintenance-xgboost",
    lstm_registered_model_name: str = "predictive-maintenance-lstm",
) -> dict[str, Any]:
    data_path = Path(data_dir)
    enable_mlflow = not no_mlflow
    allow_calibration = not no_calibration
    log = logging.getLogger("pipeline_orchestrator")
    log.info("Starting local training flow")

    xgb = _run_xgboost_step(
        data_dir=data_path,
        enable_mlflow=enable_mlflow,
        register_model=register_models,
        registered_model_name=xgb_registered_model_name,
    )
    lstm = _run_lstm_step(
        data_dir=data_path,
        sequence_length=sequence_length,
        top_k_features=top_k_features,
        epochs=epochs,
        batch_size=batch_size,
        enable_mlflow=enable_mlflow,
        register_model=register_models,
        registered_model_name=lstm_registered_model_name,
    )
    ensemble = _run_ensemble_step(
        data_dir=data_path,
        min_f2_gain=min_f2_gain,
        allow_calibration=allow_calibration,
        enable_mlflow=enable_mlflow,
    )
    log.info(
        "Local flow complete | xgb_f2=%.4f lstm_f2=%.4f selected=%s selected_f2=%.4f",
        xgb["test_f2"],
        lstm["test_f2"],
        ensemble["selected_model"],
        ensemble["test_f2"],
    )
    return {"xgboost": xgb, "lstm": lstm, "ensemble": ensemble}


def _save_summary(data_dir: Path, results: dict[str, Any], mode: str, started_at: float) -> Path:
    duration = time.time() - started_at
    summary = {
        "mode": mode,
        "duration_seconds": round(duration, 2),
        "xgboost_test_f2": float(results["xgboost"]["test_f2"]),
        "lstm_test_f2": float(results["lstm"]["test_f2"]),
        "selected_model": str(results["ensemble"]["selected_model"]),
        "selected_test_f2": float(results["ensemble"]["test_f2"]),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out_path = data_dir / "models" / "pipeline_run_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Orchestrate training pipelines with local runner or Prefect.")
    parser.add_argument("--engine", choices=["local", "prefect"], default="local", help="Execution engine")
    parser.add_argument("--data-dir", default="data", help="Path to data directory")
    parser.add_argument("--epochs", type=int, default=100, help="LSTM max epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="LSTM batch size")
    parser.add_argument("--sequence-length", type=int, default=30, help="LSTM sequence length")
    parser.add_argument("--top-k-features", type=int, default=40, help="Top features for LSTM")
    parser.add_argument("--min-f2-gain", type=float, default=0.005, help="Min val F2 gain for ensemble selection")
    parser.add_argument("--no-calibration", action="store_true", help="Disable probability calibration in ensemble")
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args()
    started = time.time()

    if args.engine == "prefect":
        if not PREFECT_AVAILABLE or training_flow_prefect is None:  # type: ignore[name-defined]
            raise RuntimeError(
                "Prefect is not installed in this environment. "
                "Install it (e.g. pip install prefect) or run with --engine local."
            )
        results = training_flow_prefect(  # type: ignore[name-defined]
            data_dir=args.data_dir,
            sequence_length=args.sequence_length,
            top_k_features=args.top_k_features,
            epochs=args.epochs,
            batch_size=args.batch_size,
            min_f2_gain=args.min_f2_gain,
            no_calibration=args.no_calibration,
            no_mlflow=args.no_mlflow,
            register_models=args.register_models,
            xgb_registered_model_name=args.xgb_registered_model_name,
            lstm_registered_model_name=args.lstm_registered_model_name,
        )
    else:
        results = training_flow_local(
            data_dir=args.data_dir,
            sequence_length=args.sequence_length,
            top_k_features=args.top_k_features,
            epochs=args.epochs,
            batch_size=args.batch_size,
            min_f2_gain=args.min_f2_gain,
            no_calibration=args.no_calibration,
            no_mlflow=args.no_mlflow,
            register_models=args.register_models,
            xgb_registered_model_name=args.xgb_registered_model_name,
            lstm_registered_model_name=args.lstm_registered_model_name,
        )

    summary_path = _save_summary(Path(args.data_dir), results, mode=args.engine, started_at=started)
    print("Pipeline orchestration complete.")
    print(f"- xgboost test_f2: {results['xgboost']['test_f2']:.4f}")
    print(f"- lstm test_f2: {results['lstm']['test_f2']:.4f}")
    print(
        f"- ensemble selected_model: {results['ensemble']['selected_model']}, "
        f"selected_test_f2: {results['ensemble']['test_f2']:.4f}"
    )
    print(f"- summary: {summary_path}")


if __name__ == "__main__":
    main()
