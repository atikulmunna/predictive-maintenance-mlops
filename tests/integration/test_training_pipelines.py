"""Integration tests for script-based training pipelines."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

from src.training.ensemble_pipeline import run_ensemble_pipeline
from src.training.lstm_pipeline import run_lstm_pipeline
from src.training.trainer import run_selected_pipeline
from src.training.xgboost_pipeline import run_xgboost_pipeline


def test_xgboost_pipeline_runs_and_saves_artifacts() -> None:
    result = run_xgboost_pipeline(data_dir=Path("data"), enable_mlflow=False)
    assert Path(result["model_path"]).exists()
    assert Path(result["scaler_path"]).exists()
    assert Path(result["feature_path"]).exists()
    assert Path(result["metrics_path"]).exists()
    assert result["test_f2"] > 0.75


def test_lstm_pipeline_runs_and_saves_artifacts() -> None:
    # Keep epochs small in tests to reduce runtime while still exercising pipeline paths.
    result = run_lstm_pipeline(
        data_dir=Path("data"),
        sequence_length=30,
        top_k_features=40,
        epochs=2,
        batch_size=64,
        enable_mlflow=False,
    )
    assert Path(result["model_path"]).exists()
    assert Path(result["scaler_path"]).exists()
    assert Path(result["features_path"]).exists()
    assert Path(result["metrics_path"]).exists()
    assert result["test_f2"] > 0.70


def test_ensemble_pipeline_runs_and_returns_policy() -> None:
    payload = run_ensemble_pipeline(
        data_dir=Path("data"),
        min_f2_gain=0.005,
        allow_calibration=True,
        enable_mlflow=False,
    )
    assert payload["selected_model"] in {"xgboost", "ensemble"}
    assert 0.0 <= payload["selected_threshold"] <= 1.0
    assert payload["test_f2"] > 0.70
    assert "confusion_matrix" in payload


def test_trainer_run_selected_pipeline_ensemble() -> None:
    args = Namespace(
        model="ensemble",
        data_dir=Path("data"),
        epochs=2,
        batch_size=64,
        sequence_length=30,
        top_k_features=40,
        min_f2_gain=0.005,
        no_calibration=False,
        no_mlflow=True,
    )
    result = run_selected_pipeline(args)
    assert "ensemble" in result
    assert result["ensemble"]["selected_model"] in {"xgboost", "ensemble"}

