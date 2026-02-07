"""FastAPI inference service for predictive maintenance models."""

from __future__ import annotations

import json
import os
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
    policy_source: str
    threshold: float
    failure_probability: float
    failure_prediction: int
    weights: dict[str, float] | None = None


class ExplainResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    equipment_id: str
    model_used_for_explanation: str
    top_k: int
    contributions: list[dict[str, Any]]


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
        self.policy_source = "artifact"

        self._apply_policy_overrides()

        self.xgb_scaler = joblib.load(self.models_dir / "scaler.pkl")
        self.lstm_scaler = joblib.load(self.models_dir / "lstm_scaler.pkl")

        self.xgb_model = xgb.XGBClassifier()
        self.xgb_model.load_model(str(self.models_dir / "xgboost_baseline.json"))
        self.lstm_model = keras.models.load_model(str(self.models_dir / "lstm_temporal.h5"))
        self._shap_explainer = None

    def _apply_policy_overrides(self) -> None:
        override = os.getenv("MODEL_OVERRIDE", "").strip().lower()
        threshold_override = os.getenv("MODEL_THRESHOLD_OVERRIDE", "").strip()

        if override:
            if override not in {"xgboost", "ensemble"}:
                raise ValueError("MODEL_OVERRIDE must be 'xgboost' or 'ensemble'")
            self.selected_model = override
            self.policy_source = "env_override"

        if threshold_override:
            try:
                threshold_val = float(threshold_override)
            except ValueError as exc:
                raise ValueError("MODEL_THRESHOLD_OVERRIDE must be a float") from exc
            if not (0.0 <= threshold_val <= 1.0):
                raise ValueError("MODEL_THRESHOLD_OVERRIDE must be between 0 and 1")
            self.threshold = threshold_val

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
            extra = [f for f in step.keys() if f not in self.xgb_features]
            if extra:
                raise HTTPException(
                    status_code=422,
                    detail=f"Unknown {len(extra)} features at timestep {idx}. Example: {extra[:5]}",
                )
            rows.append([float(step[f]) for f in self.xgb_features])
        seq = np.asarray(rows, dtype=np.float32)
        if not np.isfinite(seq).all():
            raise HTTPException(status_code=422, detail="Sequence contains non-finite values")
        return seq

    def _xgb_probability(self, sequence_arr: np.ndarray) -> float:
        last_step_df = np.array(sequence_arr[-1, :], dtype=np.float32).reshape(1, -1)
        # Scaler was fit with feature names; use DataFrame to keep column alignment.
        import pandas as pd

        last_step_df = pd.DataFrame(last_step_df, columns=self.xgb_features)
        x_scaled = self.xgb_scaler.transform(last_step_df)
        return float(self.xgb_model.predict_proba(x_scaled)[0, 1])

    def _xgb_last_step_scaled(self, sequence_arr: np.ndarray) -> np.ndarray:
        import pandas as pd

        last_step_df = pd.DataFrame(
            np.array(sequence_arr[-1, :], dtype=np.float32).reshape(1, -1),
            columns=self.xgb_features,
        )
        return self.xgb_scaler.transform(last_step_df)

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

    def explain(self, sequence: list[dict[str, float]], top_k: int = 10) -> list[dict[str, float]]:
        seq_arr = self._validate_and_convert_sequence(sequence)
        x_scaled = self._xgb_last_step_scaled(seq_arr)
        contrib = self.xgb_model.get_booster().predict(
            xgb.DMatrix(x_scaled, feature_names=self.xgb_features),
            pred_contribs=True,
        )[0]
        # Last value is bias term; exclude for feature ranking.
        feature_contrib = np.asarray(contrib[:-1], dtype=np.float32)
        pairs = sorted(
            zip(self.xgb_features, feature_contrib),
            key=lambda item: abs(float(item[1])),
            reverse=True,
        )[: max(1, top_k)]
        return [{"feature": name, "contribution": float(val)} for name, val in pairs]


def create_app(data_dir: Path | None = None) -> FastAPI:
    app = FastAPI(title="Predictive Maintenance API", version="0.1.0")
    app.state.model_bundle = None
    app.state.data_dir = data_dir or Path("data")

    @app.on_event("startup")
    def _startup() -> None:
        app.state.model_bundle = ModelBundle(app.state.data_dir)

    @app.get("/health")
    def health() -> dict[str, Any]:
        if app.state.model_bundle is None:
            raise HTTPException(status_code=503, detail="Model bundle not loaded")
        bundle: ModelBundle = app.state.model_bundle
        return {
            "status": "ok",
            "selected_model_policy": bundle.selected_model,
            "policy_source": bundle.policy_source,
            "sequence_length": bundle.sequence_length,
        }

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
            policy_source=bundle.policy_source,
            threshold=bundle.threshold,
            failure_probability=proba,
            failure_prediction=pred,
            weights=bundle.weights if bundle.selected_model == "ensemble" else None,
        )

    @app.post("/explain", response_model=ExplainResponse)
    def explain(request: PredictRequest, top_k: int = 10) -> ExplainResponse:
        bundle: ModelBundle | None = app.state.model_bundle
        if bundle is None:
            raise HTTPException(status_code=503, detail="Model bundle not loaded")
        if top_k < 1 or top_k > 50:
            raise HTTPException(status_code=422, detail="top_k must be between 1 and 50")
        contributions = bundle.explain(request.sequence, top_k=top_k)
        return ExplainResponse(
            equipment_id=request.equipment_id,
            model_used_for_explanation="xgboost",
            top_k=top_k,
            contributions=contributions,
        )

    return app


app = create_app()
