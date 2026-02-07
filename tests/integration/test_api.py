"""Integration smoke tests for FastAPI serving endpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

# Ensure local `src` package is importable in pytest contexts.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.serving.api import create_app


def _build_valid_sequence() -> list[dict[str, float]]:
    models_dir = Path("data") / "models"
    with open(models_dir / "feature_names.json", "r", encoding="utf-8") as f:
        xgb_features = json.load(f)["features"]
    with open(models_dir / "lstm_features.json", "r", encoding="utf-8") as f:
        sequence_length = int(json.load(f).get("sequence_length", 30))
    step = {name: 0.0 for name in xgb_features}
    return [step.copy() for _ in range(sequence_length)]


def test_health_endpoint() -> None:
    app = create_app(data_dir=Path("data"))
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_smoke() -> None:
    app = create_app(data_dir=Path("data"))
    payload = {"equipment_id": "engine_smoke_001", "sequence": _build_valid_sequence()}
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert body["equipment_id"] == "engine_smoke_001"
    assert body["model_used"] in {"xgboost", "ensemble"}
    assert body["selected_model_policy"] in {"xgboost", "ensemble"}
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["failure_prediction"] in {0, 1}
    assert 0.0 <= body["threshold"] <= 1.0
