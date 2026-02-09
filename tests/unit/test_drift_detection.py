"""Unit tests for drift detection module."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from src.monitoring import drift_detection


def _write_csv(path: Path, rows: list[dict[str, float]]) -> None:
    pd.DataFrame(rows).to_csv(path, index=False)


def test_detect_feature_drift_stable_case() -> None:
    ref = pd.DataFrame({"a": [0.1, 0.2, 0.3, 0.4], "b": [1.0, 1.1, 1.2, 1.3]})
    cur = ref.copy()
    out = drift_detection.detect_feature_drift(ref, cur, ["a", "b"], p_value_threshold=0.001, psi_threshold=0.5)
    assert out["feature_count_checked"] == 2
    assert out["drifted_feature_count"] == 0


def test_detect_prediction_drift_when_shifted() -> None:
    ref = pd.DataFrame({"failure_probability": [0.1, 0.1, 0.2, 0.2, 0.3]})
    cur = pd.DataFrame({"failure_probability": [0.8, 0.85, 0.9, 0.95, 0.9]})
    out = drift_detection.detect_prediction_drift(
        ref,
        cur,
        "failure_probability",
        p_value_threshold=0.05,
        mean_shift_threshold=0.1,
    )
    assert out is not None
    assert out["is_drift"] is True
    assert out["mean_shift_abs"] >= 0.1


def test_run_drift_detection_writes_report(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "models").mkdir(parents=True, exist_ok=True)
    ref_path = tmp_path / "reference.csv"
    cur_path = tmp_path / "current.csv"

    _write_csv(
        ref_path,
        [
            {"f1": 0.1, "f2": 1.0, "failure_probability": 0.15},
            {"f1": 0.2, "f2": 1.1, "failure_probability": 0.20},
            {"f1": 0.3, "f2": 1.2, "failure_probability": 0.25},
            {"f1": 0.4, "f2": 1.3, "failure_probability": 0.30},
        ],
    )
    _write_csv(
        cur_path,
        [
            {"f1": 1.1, "f2": 3.0, "failure_probability": 0.85},
            {"f1": 1.2, "f2": 3.1, "failure_probability": 0.90},
            {"f1": 1.3, "f2": 3.2, "failure_probability": 0.95},
            {"f1": 1.4, "f2": 3.3, "failure_probability": 0.99},
        ],
    )

    feature_names = {"features": ["f1", "f2"]}
    (data_dir / "models" / "feature_names.json").write_text(json.dumps(feature_names), encoding="utf-8")

    out = drift_detection.run_drift_detection(reference_path=ref_path, current_path=cur_path, data_dir=data_dir)
    assert out["summary"]["drift_detected"] is True
    report_path = data_dir / "models" / "drift_report.json"
    assert report_path.exists()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["summary"]["status"] == "drift_detected"


def test_main_cli_runs(monkeypatch, tmp_path: Path, capsys) -> None:
    ref_path = tmp_path / "reference.csv"
    cur_path = tmp_path / "current.csv"
    _write_csv(ref_path, [{"f1": 0.1}, {"f1": 0.2}, {"f1": 0.3}])
    _write_csv(cur_path, [{"f1": 0.1}, {"f1": 0.21}, {"f1": 0.31}])
    out_path = tmp_path / "drift.json"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--reference",
            str(ref_path),
            "--current",
            str(cur_path),
            "--output",
            str(out_path),
        ],
    )
    drift_detection.main()
    out = capsys.readouterr().out
    assert "Drift detection complete." in out
    assert out_path.exists()
