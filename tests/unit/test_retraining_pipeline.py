"""Unit tests for conditional retraining pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from src.pipelines import retraining_pipeline


def _write_drift_report(path: Path, *, drift: bool, feature_count: int, pred_drift: bool) -> None:
    payload = {
        "summary": {
            "drift_detected": drift,
            "feature_drifted_count": feature_count,
            "prediction_drift_detected": pred_drift,
        }
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_should_trigger_retraining_rules() -> None:
    trigger, reason = retraining_pipeline.should_trigger_retraining(None)
    assert trigger is False
    assert reason == "drift_report_missing_or_invalid"

    report = {"summary": {"drift_detected": False, "feature_drifted_count": 10, "prediction_drift_detected": True}}
    trigger, reason = retraining_pipeline.should_trigger_retraining(report)
    assert trigger is False
    assert reason == "drift_not_detected"

    report2 = {"summary": {"drift_detected": True, "feature_drifted_count": 1, "prediction_drift_detected": True}}
    trigger, reason = retraining_pipeline.should_trigger_retraining(report2, min_drifted_features=3)
    assert trigger is False
    assert reason == "feature_drift_below_threshold"

    report3 = {"summary": {"drift_detected": True, "feature_drifted_count": 4, "prediction_drift_detected": False}}
    trigger, reason = retraining_pipeline.should_trigger_retraining(report3, require_prediction_drift=True)
    assert trigger is False
    assert reason == "prediction_drift_required_not_met"

    report4 = {"summary": {"drift_detected": True, "feature_drifted_count": 4, "prediction_drift_detected": True}}
    trigger, reason = retraining_pipeline.should_trigger_retraining(report4, require_prediction_drift=True)
    assert trigger is True
    assert reason == "drift_policy_triggered"


def test_run_retraining_decision_skips_when_not_triggered(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    report_path = data_dir / "models" / "drift_report.json"
    _write_drift_report(report_path, drift=False, feature_count=0, pred_drift=False)

    out = retraining_pipeline.run_retraining_decision(data_dir=data_dir)
    assert out["retraining_triggered"] is False
    assert out["reason"] == "drift_not_detected"
    assert (data_dir / "models" / "retraining_decision.json").exists()
    assert out["results"] is None


def test_run_retraining_decision_triggers_local(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    report_path = data_dir / "models" / "drift_report.json"
    _write_drift_report(report_path, drift=True, feature_count=5, pred_drift=False)

    monkeypatch.setattr(
        retraining_pipeline.prefect_flow,
        "training_flow_local",
        lambda **kwargs: {  # type: ignore[no-untyped-def]
            "xgboost": {"test_f2": 0.9},
            "lstm": {"test_f2": 0.8},
            "ensemble": {"selected_model": "xgboost", "test_f2": 0.88},
        },
    )
    out = retraining_pipeline.run_retraining_decision(data_dir=data_dir, no_mlflow=True)
    assert out["retraining_triggered"] is True
    assert out["reason"] == "drift_policy_triggered"
    assert out["results"] is not None
    assert out["results"]["ensemble"]["selected_model"] == "xgboost"


def test_run_retraining_decision_force_triggers(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "models").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        retraining_pipeline.prefect_flow,
        "training_flow_local",
        lambda **kwargs: {"xgboost": {"test_f2": 0.9}, "lstm": {"test_f2": 0.8}, "ensemble": {"selected_model": "xgboost", "test_f2": 0.88}},  # type: ignore[no-untyped-def]
    )
    out = retraining_pipeline.run_retraining_decision(data_dir=data_dir, force=True, no_mlflow=True)
    assert out["retraining_triggered"] is True
    assert out["reason"] == "forced"


def test_run_retraining_prefect_unavailable_raises(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    report_path = data_dir / "models" / "drift_report.json"
    _write_drift_report(report_path, drift=True, feature_count=5, pred_drift=True)

    monkeypatch.setattr(retraining_pipeline.prefect_flow, "PREFECT_AVAILABLE", False)
    with pytest.raises(RuntimeError, match="Prefect is not installed"):
        retraining_pipeline.run_retraining_decision(data_dir=data_dir, engine="prefect")


def test_main_cli_prints_summary(monkeypatch, tmp_path: Path, capsys) -> None:
    data_dir = tmp_path / "data"
    report_path = data_dir / "models" / "drift_report.json"
    _write_drift_report(report_path, drift=False, feature_count=0, pred_drift=False)

    monkeypatch.setattr(sys, "argv", ["prog", "--data-dir", str(data_dir)])
    retraining_pipeline.main()
    out = capsys.readouterr().out
    assert "Retraining decision complete." in out
    assert "triggered: False" in out
