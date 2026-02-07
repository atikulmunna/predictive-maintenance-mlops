"""Utilities for optional MLflow logging in script-based pipelines."""

from __future__ import annotations

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

