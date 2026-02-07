"""Unit tests for CLI main() functions in training modules."""

from __future__ import annotations

import sys
from pathlib import Path

from src.training import ensemble_pipeline, lstm_pipeline, trainer, xgboost_pipeline


def test_xgboost_main_prints_summary(monkeypatch, capsys) -> None:
    fake_result = {
        "model_path": "data/models/xgboost_baseline.json",
        "scaler_path": "data/models/scaler.pkl",
        "feature_path": "data/models/feature_names.json",
        "metrics_path": "data/models/xgboost_baseline_metrics.json",
        "test_f2": 0.9,
        "test_precision": 0.8,
        "test_recall": 0.95,
        "test_roc_auc": 0.99,
        "mlflow_run_id": "run-xgb-1",
    }
    monkeypatch.setattr(xgboost_pipeline, "run_xgboost_pipeline", lambda *a, **k: fake_result)
    monkeypatch.setattr(sys, "argv", ["prog", "--data-dir", "data", "--no-mlflow"])
    xgboost_pipeline.main()
    out = capsys.readouterr().out
    assert "Saved model:" in out
    assert "MLflow run:" in out


def test_lstm_main_prints_summary(monkeypatch, capsys) -> None:
    fake_result = {
        "model_path": "data/models/lstm_temporal.h5",
        "scaler_path": "data/models/lstm_scaler.pkl",
        "features_path": "data/models/lstm_features.json",
        "metrics_path": "data/models/lstm_temporal_metrics.json",
        "test_f2": 0.85,
        "test_precision": 0.8,
        "test_recall": 0.9,
        "test_roc_auc": 0.98,
        "epochs_trained": 4,
        "mlflow_run_id": "run-lstm-1",
    }
    monkeypatch.setattr(lstm_pipeline, "run_lstm_pipeline", lambda *a, **k: fake_result)
    monkeypatch.setattr(sys, "argv", ["prog", "--data-dir", "data", "--epochs", "2", "--no-mlflow"])
    lstm_pipeline.main()
    out = capsys.readouterr().out
    assert "Saved model:" in out
    assert "Epochs trained:" in out
    assert "MLflow run:" in out


def test_ensemble_main_writes_metrics(monkeypatch, tmp_path, capsys) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "models").mkdir(parents=True, exist_ok=True)
    fake_payload = {
        "selected_model": "xgboost",
        "test_f2": 0.99,
        "val_f2_ensemble_best": 0.99,
    }
    monkeypatch.setattr(ensemble_pipeline, "run_ensemble_pipeline", lambda *a, **k: fake_payload)
    monkeypatch.setattr(sys, "argv", ["prog", "--data-dir", str(data_dir), "--no-mlflow"])
    ensemble_pipeline.main()
    out = capsys.readouterr().out
    assert "Saved:" in out
    assert "Selected model:" in out
    assert (data_dir / "models" / "ensemble_metrics.json").exists()


def test_trainer_main_prints_pipeline_summary(monkeypatch, capsys) -> None:
    fake = {"ensemble": {"selected_model": "xgboost", "test_f2": 0.99}}
    monkeypatch.setattr(trainer, "run_selected_pipeline", lambda *a, **k: fake)
    monkeypatch.setattr(sys, "argv", ["prog", "--model", "ensemble", "--data-dir", "data", "--no-mlflow"])
    trainer.main()
    out = capsys.readouterr().out
    assert "Pipeline run complete." in out
    assert "selected_model=xgboost" in out

