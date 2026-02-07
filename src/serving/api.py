"""FastAPI inference service for predictive maintenance models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from tensorflow import keras


class PredictRequest(BaseModel):
    equipment_id: str = Field(..., description="Unique equipment identifier")
    sequence: list[dict[str, float]] = Field(
        ..., description="Time-ordered list of feature snapshots; must be sequence_length long"
    )


class PredictResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    equipment_id: str
    model_used: str
    selected_model_policy: str
    threshold: float
    failure_probability: float
    failure_prediction: int
    weights: dict[str, float] | None = None


class ModelBundle:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.models_dir = self.data_dir / "models"
        self._load()

    def _load(self) -> None:
        if not self.models_dir.exists():
            raise FileNotFoundError(f"Models directory not found: {self.models_dir}")

        with open(self.models_dir / "feature_names.json", "r", encoding="utf-8") as f:
            self.xgb_features = json.load(f)["features"]

        with open(self.models_dir / "lstm_features.json", "r", encoding="utf-8") as f:
            lstm_meta = json.load(f)
        self.lstm_features = lstm_meta["features"]
        self.sequence_length = int(lstm_meta.get("sequence_length", 30))

        with open(self.models_dir / "ensemble_metrics.json", "r", encoding="utf-8") as f:
            self.ensemble_metrics = json.load(f)

        self.selected_model = str(self.ensemble_metrics.get("selected_model", "xgboost"))
        self.threshold = float(self.ensemble_metrics.get("selected_threshold", 0.5))
        self.weights = self.ensemble_metrics.get("weights", {"xgboost": 1.0, "lstm": 0.0})

        self.xgb_scaler = joblib.load(self.models_dir / "scaler.pkl")
        self.lstm_scaler = joblib.load(self.models_dir / "lstm_scaler.pkl")

        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(str(self.models_dir / "xgboost_baseline.json"))
        self.lstm_model = keras.models.load_model(str(self.models_dir / "lstm_temporal.h5"))

    def _validate_and_convert_sequence(self, sequence: list[dict[str, float]]) -> np.ndarray:
        if len(sequence) != self.sequence_length:
            raise HTTPException(
                status_code=422,
                detail=f"Expected sequence length {self.sequence_length}, got {len(sequence)}",
            )
        rows: list[list[float]] = []
        for idx, step in enumerate(sequence):
            missing = [f for f in self.xgb_features if f not in step]
            if missing:
                raise HTTPException(
                    status_code=422,
                    detail=f"Missing {len(missing)} features at timestep {idx}. Example: {missing[:5]}",
                )
            rows.append([float(step[f]) for f in self.xgb_features])
        return np.asarray(rows, dtype=np.float32)

    def _xgb_probability(self, sequence_arr: np.ndarray) -> float:
        last_step_df = np.array(sequence_arr[-1, :], dtype=np.float32).reshape(1, -1)
        # Scaler was fit with feature names; use DataFrame to keep column alignment.
        import pandas as pd

        last_step_df = pd.DataFrame(last_step_df, columns=self.xgb_features)
        x_scaled = self.xgb_scaler.transform(last_step_df)
        return float(self.xgb_model.predict_proba(x_scaled)[0, 1])

    def _lstm_probability(self, sequence_arr: np.ndarray) -> float:
        idx = [self.xgb_features.index(f) for f in self.lstm_features]
        lstm_seq = sequence_arr[:, idx]
        n_steps, n_feat = lstm_seq.shape
        lstm_scaled = self.lstm_scaler.transform(lstm_seq.reshape(-1, n_feat)).reshape(1, n_steps, n_feat)
        return float(self.lstm_model.predict(lstm_scaled, verbose=0).flatten()[0])

    def predict(self, sequence: list[dict[str, float]]) -> tuple[float, int, str]:
        seq_arr = self._validate_and_convert_sequence(sequence)
        xgb_proba = self._xgb_probability(seq_arr)
        lstm_proba = self._lstm_probability(seq_arr)

        if self.selected_model == "ensemble":
            w_xgb = float(self.weights.get("xgboost", 0.5))
            w_lstm = float(self.weights.get("lstm", 0.5))
            proba = (w_xgb * xgb_proba) + (w_lstm * lstm_proba)
            model_used = "ensemble"
        else:
            proba = xgb_proba
            model_used = "xgboost"

        pred = int(proba >= self.threshold)
        return float(proba), pred, model_used


def create_app(data_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="Predictive Maintenance API", version="0.1.0")
    app.state.model_bundle = None
    app.state.data_dir = data_dir or Path("data")

    @app.on_event("startup")
    def _startup() -> None:
        app.state.model_bundle = ModelBundle(app.state.data_dir)

    @app.get("/health")
    def health() -> dict[str, str]:
        if app.state.model_bundle is None:
            raise HTTPException(status_code=503, detail="Model bundle not loaded")
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        bundle: ModelBundle | None = app.state.model_bundle
        if bundle is None:
            raise HTTPException(status_code=503, detail="Model bundle not loaded")
        proba, pred, model_used = bundle.predict(request.sequence)
        return PredictResponse(
            equipment_id=request.equipment_id,
            model_used=model_used,
            selected_model_policy=bundle.selected_model,
            threshold=bundle.threshold,
            failure_probability=proba,
            failure_prediction=pred,
            weights=bundle.weights if bundle.selected_model == "ensemble" else None,
        )

    return app


app = create_app()
