"""Integration smoke tests for FastAPI serving endpoints."""

from __future__ import annotations

import json
import os
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
    body = response.json()
    assert body["status"] == "ok"
    assert body["selected_model_policy"] in {"xgboost", "ensemble"}
    assert body["policy_source"] in {"artifact", "env_override"}
    assert body["sequence_length"] == 30


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
    assert body["policy_source"] in {"artifact", "env_override"}
    assert 0.0 <= body["failure_probability"] <= 1.0
    assert body["failure_prediction"] in {0, 1}
    assert 0.0 <= body["threshold"] <= 1.0


def test_predict_rejects_missing_feature() -> None:
    sequence = _build_valid_sequence()
    sequence[0].pop(next(iter(sequence[0].keys())))
    app = create_app(data_dir=Path("data"))
    payload = {"equipment_id": "engine_missing_001", "sequence": sequence}
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "Missing" in response.json()["detail"]


def test_predict_rejects_unknown_feature() -> None:
    sequence = _build_valid_sequence()
    sequence[0]["unknown_feature_foo"] = 1.0
    app = create_app(data_dir=Path("data"))
    payload = {"equipment_id": "engine_extra_001", "sequence": sequence}
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "Unknown" in response.json()["detail"]


def test_predict_rejects_non_finite_values() -> None:
    sequence = _build_valid_sequence()
    first_key = next(iter(sequence[0].keys()))
    sequence[0][first_key] = float("nan")
    app = create_app(data_dir=Path("data"))
    payload = {"equipment_id": "engine_nan_001", "sequence": sequence}
    with TestClient(app) as client:
        response = client.post("/predict", json=payload)
    assert response.status_code == 422
    assert "non-finite" in response.json()["detail"]


def test_model_override_env_policy() -> None:
    old_override = os.environ.get("MODEL_OVERRIDE")
    try:
        os.environ["MODEL_OVERRIDE"] = "xgboost"
        app = create_app(data_dir=Path("data"))
        payload = {"equipment_id": "engine_override_001", "sequence": _build_valid_sequence()}
        with TestClient(app) as client:
            response = client.post("/predict", json=payload)
        assert response.status_code == 200
        body = response.json()
        assert body["selected_model_policy"] == "xgboost"
        assert body["policy_source"] == "env_override"
    finally:
        if old_override is None:
            os.environ.pop("MODEL_OVERRIDE", None)
        else:
            os.environ["MODEL_OVERRIDE"] = old_override


def test_explain_endpoint_smoke() -> None:
    app = create_app(data_dir=Path("data"))
    payload = {"equipment_id": "engine_explain_001", "sequence": _build_valid_sequence()}
    with TestClient(app) as client:
        response = client.post("/explain?top_k=5", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["equipment_id"] == "engine_explain_001"
    assert body["model_used_for_explanation"] == "xgboost"
    assert body["top_k"] == 5
    assert isinstance(body["contributions"], list)
    assert len(body["contributions"]) == 5
    for item in body["contributions"]:
        assert "feature" in item
        assert "contribution" in item
