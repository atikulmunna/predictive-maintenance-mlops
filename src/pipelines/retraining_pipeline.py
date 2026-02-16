"""Conditional retraining pipeline driven by drift report state."""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

from src.pipelines import prefect_flow


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _decision_path(data_dir: Path) -> Path:
    return data_dir / "models" / "retraining_decision.json"


def _report_path(data_dir: Path, report_path: Path | None) -> Path:
    return report_path if report_path is not None else (data_dir / "models" / "drift_report.json")


def should_trigger_retraining(
    report: dict[str, Any] | None,
    *,
    min_drifted_features: int = 3,
    require_prediction_drift: bool = False,
) -> tuple[bool, str]:
    if report is None:
        return False, "drift_report_missing_or_invalid"

    summary = report.get("summary", {})
    drift_detected = bool(summary.get("drift_detected", False))
    feature_count = int(summary.get("feature_drifted_count", 0) or 0)
    prediction_drift = bool(summary.get("prediction_drift_detected", False))

    if not drift_detected:
        return False, "drift_not_detected"
    if feature_count < min_drifted_features:
        return False, "feature_drift_below_threshold"
    if require_prediction_drift and not prediction_drift:
        return False, "prediction_drift_required_not_met"
    return True, "drift_policy_triggered"


def run_retraining_decision(
    *,
    data_dir: Path,
    engine: str = "local",
    report_path: Path | None = None,
    force: bool = False,
    min_drifted_features: int = 3,
    require_prediction_drift: bool = False,
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
    started = time.time()
    resolved_report_path = _report_path(data_dir, report_path)
    report = _load_json(resolved_report_path)
    trigger, reason = should_trigger_retraining(
        report,
        min_drifted_features=min_drifted_features,
        require_prediction_drift=require_prediction_drift,
    )
    if force:
        trigger = True
        reason = "forced"

    result_payload: dict[str, Any] | None = None
    if trigger:
        if engine == "prefect":
            if not prefect_flow.PREFECT_AVAILABLE or getattr(prefect_flow, "training_flow_prefect", None) is None:
                raise RuntimeError(
                    "Prefect is not installed in this environment. Install it or use --engine local."
                )
            result_payload = prefect_flow.training_flow_prefect(  # type: ignore[attr-defined]
                data_dir=str(data_dir),
                sequence_length=sequence_length,
                top_k_features=top_k_features,
                epochs=epochs,
                batch_size=batch_size,
                min_f2_gain=min_f2_gain,
                no_calibration=no_calibration,
                no_mlflow=no_mlflow,
                register_models=register_models,
                xgb_registered_model_name=xgb_registered_model_name,
                lstm_registered_model_name=lstm_registered_model_name,
            )
        else:
            result_payload = prefect_flow.training_flow_local(
                data_dir=str(data_dir),
                sequence_length=sequence_length,
                top_k_features=top_k_features,
                epochs=epochs,
                batch_size=batch_size,
                min_f2_gain=min_f2_gain,
                no_calibration=no_calibration,
                no_mlflow=no_mlflow,
                register_models=register_models,
                xgb_registered_model_name=xgb_registered_model_name,
                lstm_registered_model_name=lstm_registered_model_name,
            )

    decision = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "engine": engine,
        "drift_report_path": str(resolved_report_path),
        "retraining_triggered": bool(trigger),
        "reason": reason,
        "policy": {
            "min_drifted_features": min_drifted_features,
            "require_prediction_drift": require_prediction_drift,
            "forced": force,
        },
        "duration_seconds": round(time.time() - started, 2),
        "results": result_payload,
    }
    out_path = _decision_path(data_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(decision, indent=2), encoding="utf-8")
    decision["output_path"] = str(out_path)
    return decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run conditional retraining based on drift report.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Path to data directory")
    parser.add_argument("--engine", choices=["local", "prefect"], default="local", help="Execution engine")
    parser.add_argument("--drift-report", type=Path, default=None, help="Override drift report path")
    parser.add_argument("--force", action="store_true", help="Force retraining regardless of drift state")
    parser.add_argument("--min-drifted-features", type=int, default=3, help="Feature drift threshold to trigger")
    parser.add_argument(
        "--require-prediction-drift",
        action="store_true",
        help="Require prediction drift in addition to feature drift",
    )
    parser.add_argument("--epochs", type=int, default=100, help="LSTM max epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="LSTM batch size")
    parser.add_argument("--sequence-length", type=int, default=30, help="LSTM sequence length")
    parser.add_argument("--top-k-features", type=int, default=40, help="Top features for LSTM")
    parser.add_argument("--min-f2-gain", type=float, default=0.005, help="Min val F2 gain for ensemble selection")
    parser.add_argument("--no-calibration", action="store_true", help="Disable probability calibration")
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
    payload = run_retraining_decision(
        data_dir=args.data_dir,
        engine=args.engine,
        report_path=args.drift_report,
        force=args.force,
        min_drifted_features=args.min_drifted_features,
        require_prediction_drift=args.require_prediction_drift,
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
    print("Retraining decision complete.")
    print(f"- triggered: {payload['retraining_triggered']}")
    print(f"- reason: {payload['reason']}")
    print(f"- report: {payload['drift_report_path']}")
    print(f"- output: {payload['output_path']}")


if __name__ == "__main__":
    main()
