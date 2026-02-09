"""Utilities for optional MLflow logging in script-based pipelines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def get_mlflow_client(data_dir: Path, experiment_name: str) -> Any | None:
    """Return configured mlflow module, or None if unavailable."""
    try:
        import mlflow
    except Exception:
        return None

    tracking_uri = f"file:{(data_dir.parent / 'mlruns').resolve()}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return mlflow


def log_artifacts_if_exist(mlflow_client: Any, paths: list[Path]) -> None:
    for p in paths:
        if p.exists():
            mlflow_client.log_artifact(str(p))


def register_model_if_possible(
    mlflow_client: Any | None,
    run_id: str | None,
    artifact_path: str,
    registered_model_name: str,
) -> dict[str, Any]:
    """Try to register a model version in MLflow registry.

    Returns a status payload; never raises.
    """
    if mlflow_client is None:
        return {"status": "skipped", "reason": "mlflow_unavailable"}
    if not run_id:
        return {"status": "skipped", "reason": "missing_run_id"}
    try:
        model_uri = f"runs:/{run_id}/{artifact_path}"
        model_version = mlflow_client.register_model(model_uri=model_uri, name=registered_model_name)
        version = getattr(model_version, "version", None)
        return {
            "status": "registered",
            "registered_model_name": registered_model_name,
            "model_uri": model_uri,
            "version": str(version) if version is not None else None,
        }
    except Exception as exc:  # pragma: no cover - backend-dependent behavior
        return {
            "status": "failed",
            "registered_model_name": registered_model_name,
            "error": str(exc),
        }


def write_registry_state(data_dir: Path, payload: dict[str, Any]) -> Path:
    """Persist local model-registry state for promotion decisions."""
    models_dir = data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / "model_registry_state.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path
