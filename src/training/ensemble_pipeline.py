"""Validation-tuned ensemble pipeline for XGBoost + LSTM artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, fbeta_score, precision_score, recall_score, roc_auc_score
from tensorflow import keras


EXCLUDE_COLS = ["unit_id", "cycle", "RUL", "failure_soon"]


@dataclass
class EnsembleInputs:
    x_val_flat: np.ndarray
    x_test_flat: np.ndarray
    x_val_lstm: np.ndarray
    x_test_lstm: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def create_sequences(df: pd.DataFrame, feature_cols: list[str], sequence_length: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_seq: list[np.ndarray] = []
    y_seq: list[float] = []
    engine_ids: list[Any] = []
    for engine_id in df["unit_id"].unique():
        engine_df = df[df["unit_id"] == engine_id]
        x = engine_df[feature_cols].values
        y = engine_df["failure_soon"].values
        for i in range(len(x) - sequence_length + 1):
            x_seq.append(x[i : i + sequence_length, :])
            y_seq.append(y[i + sequence_length - 1])
            engine_ids.append(engine_id)
    return np.array(x_seq), np.array(y_seq), np.array(engine_ids)


def split_by_engine(engine_ids: np.ndarray, unique_engines: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_engines = len(unique_engines)
    n_train = int(0.70 * n_engines)
    n_val = int(0.15 * n_engines)

    train_engines = np.sort(unique_engines[:n_train])
    val_engines = np.sort(unique_engines[n_train : n_train + n_val])
    test_engines = np.sort(unique_engines[n_train + n_val :])

    train_mask = np.isin(engine_ids, train_engines)
    val_mask = np.isin(engine_ids, val_engines)
    test_mask = np.isin(engine_ids, test_engines)
    return train_mask, val_mask, test_mask


def best_f2_threshold(y_true: np.ndarray, probas: np.ndarray, beta: int = 2) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t = 0.5
    best_score = -1.0
    for t in thresholds:
        pred = (probas >= t).astype(int)
        score = fbeta_score(y_true, pred, beta=beta)
        if score > best_score:
            best_t = float(t)
            best_score = float(score)
    return best_t, best_score


def platt_scale(val_proba: np.ndarray, y_val: np.ndarray, test_proba: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    val_proba = np.clip(val_proba, 1e-6, 1 - 1e-6)
    test_proba = np.clip(test_proba, 1e-6, 1 - 1e-6)
    lr = LogisticRegression(solver="lbfgs", max_iter=1000)
    lr.fit(val_proba.reshape(-1, 1), y_val)
    val_cal = lr.predict_proba(val_proba.reshape(-1, 1))[:, 1]
    test_cal = lr.predict_proba(test_proba.reshape(-1, 1))[:, 1]
    return val_cal, test_cal


def prepare_inputs(data_dir: Path) -> tuple[EnsembleInputs, dict[str, Any]]:
    models_dir = data_dir / "models"
    processed_dir = data_dir / "processed"

    df = pd.read_csv(processed_dir / "train_features_FD001.csv")
    with open(models_dir / "feature_names.json", "r", encoding="utf-8") as f:
        xgb_features = json.load(f)["features"]
    with open(models_dir / "lstm_features.json", "r", encoding="utf-8") as f:
        lstm_artifact = json.load(f)
    lstm_features = lstm_artifact["features"]
    sequence_length = int(lstm_artifact.get("sequence_length", 30))

    x_seq, y_seq, engine_ids = create_sequences(df, xgb_features, sequence_length)
    train_mask, val_mask, test_mask = split_by_engine(engine_ids, df["unit_id"].unique())

    x_train_seq = x_seq[train_mask]
    x_val_seq = x_seq[val_mask]
    x_test_seq = x_seq[test_mask]
    y_val = y_seq[val_mask]
    y_test = y_seq[test_mask]

    xgb_scaler = joblib.load(models_dir / "scaler.pkl")
    x_val_last_df = pd.DataFrame(x_val_seq[:, -1, :], columns=xgb_features)
    x_test_last_df = pd.DataFrame(x_test_seq[:, -1, :], columns=xgb_features)
    x_val_flat = xgb_scaler.transform(x_val_last_df)
    x_test_flat = xgb_scaler.transform(x_test_last_df)

    lstm_indices = [xgb_features.index(f) for f in lstm_features]
    x_train_lstm = x_train_seq[:, :, lstm_indices]
    x_val_lstm = x_val_seq[:, :, lstm_indices]
    x_test_lstm = x_test_seq[:, :, lstm_indices]

    lstm_scaler = joblib.load(models_dir / "lstm_scaler.pkl")
    n_val, t_val, f_val = x_val_lstm.shape
    n_test, t_test, f_test = x_test_lstm.shape
    x_val_lstm_scaled = lstm_scaler.transform(x_val_lstm.reshape(-1, f_val)).reshape(n_val, t_val, f_val)
    x_test_lstm_scaled = lstm_scaler.transform(x_test_lstm.reshape(-1, f_test)).reshape(n_test, t_test, f_test)

    ctx = {
        "models_dir": models_dir,
        "xgb_features": xgb_features,
        "lstm_features": lstm_features,
        "sequence_length": sequence_length,
    }
    return (
        EnsembleInputs(
            x_val_flat=x_val_flat,
            x_test_flat=x_test_flat,
            x_val_lstm=x_val_lstm_scaled,
            x_test_lstm=x_test_lstm_scaled,
            y_val=y_val,
            y_test=y_test,
        ),
        ctx,
    )


def run_ensemble_pipeline(data_dir: Path, min_f2_gain: float = 0.005, allow_calibration: bool = True) -> dict[str, Any]:
    inputs, ctx = prepare_inputs(data_dir)
    models_dir = ctx["models_dir"]

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(str(models_dir / "xgboost_baseline.json"))
    lstm_model = keras.models.load_model(str(models_dir / "lstm_temporal.h5"))

    with open(models_dir / "xgboost_baseline_metrics.json", "r", encoding="utf-8") as f:
        xgb_metrics_all = json.load(f)
    with open(models_dir / "lstm_temporal_metrics.json", "r", encoding="utf-8") as f:
        lstm_metrics_all = json.load(f)

    xgb_val_ref = float(xgb_metrics_all["validation_metrics"]["f2_score"])
    xgb_test = xgb_metrics_all["test_metrics"]
    lstm_test_metrics = lstm_metrics_all["test_metrics"]

    xgb_val_raw = xgb_model.predict_proba(inputs.x_val_flat)[:, 1]
    xgb_test_raw = xgb_model.predict_proba(inputs.x_test_flat)[:, 1]
    lstm_val_raw = lstm_model.predict(inputs.x_val_lstm, verbose=0).flatten()
    lstm_test_raw = lstm_model.predict(inputs.x_test_lstm, verbose=0).flatten()

    if allow_calibration:
        xgb_val_cal, xgb_test_cal = platt_scale(xgb_val_raw, inputs.y_val, xgb_test_raw)
        lstm_val_cal, lstm_test_cal = platt_scale(lstm_val_raw, inputs.y_val, lstm_test_raw)
    else:
        xgb_val_cal, xgb_test_cal = xgb_val_raw, xgb_test_raw
        lstm_val_cal, lstm_test_cal = lstm_val_raw, lstm_test_raw

    xgb_t_raw, xgb_f2_raw = best_f2_threshold(inputs.y_val, xgb_val_raw)
    xgb_t_cal, xgb_f2_cal = best_f2_threshold(inputs.y_val, xgb_val_cal)
    if xgb_f2_cal > xgb_f2_raw:
        xgb_variant = "calibrated"
        xgb_val, xgb_test = xgb_val_cal, xgb_test_cal
        xgb_t_best, xgb_f2_best = xgb_t_cal, xgb_f2_cal
    else:
        xgb_variant = "raw"
        xgb_val, xgb_test = xgb_val_raw, xgb_test_raw
        xgb_t_best, xgb_f2_best = xgb_t_raw, xgb_f2_raw

    lstm_t_raw, lstm_f2_raw = best_f2_threshold(inputs.y_val, lstm_val_raw)
    lstm_t_cal, lstm_f2_cal = best_f2_threshold(inputs.y_val, lstm_val_cal)
    if lstm_f2_cal > lstm_f2_raw:
        lstm_variant = "calibrated"
        lstm_val, lstm_test = lstm_val_cal, lstm_test_cal
    else:
        lstm_variant = "raw"
        lstm_val, lstm_test = lstm_val_raw, lstm_test_raw

    best_weight = 0.6
    best_threshold = 0.5
    best_val_f2 = -1.0
    best_test_proba = None
    for w in np.linspace(0.0, 1.0, 41):
        val_proba = (w * xgb_val) + ((1.0 - w) * lstm_val)
        test_proba = (w * xgb_test) + ((1.0 - w) * lstm_test)
        t, f2 = best_f2_threshold(inputs.y_val, val_proba)
        if f2 > best_val_f2:
            best_weight = float(w)
            best_threshold = float(t)
            best_val_f2 = float(f2)
            best_test_proba = test_proba

    if best_test_proba is None:
        raise RuntimeError("Ensemble search failed to produce probabilities")

    ensemble_pred = (best_test_proba >= best_threshold).astype(int)
    ensemble_f2 = float(fbeta_score(inputs.y_test, ensemble_pred, beta=2))
    ensemble_precision = float(precision_score(inputs.y_test, ensemble_pred))
    ensemble_recall = float(recall_score(inputs.y_test, ensemble_pred))
    ensemble_auc = float(roc_auc_score(inputs.y_test, best_test_proba))

    xgb_pred = (xgb_test >= xgb_t_best).astype(int)
    xgb_f2_test_tuned = float(fbeta_score(inputs.y_test, xgb_pred, beta=2))
    xgb_precision_tuned = float(precision_score(inputs.y_test, xgb_pred))
    xgb_recall_tuned = float(recall_score(inputs.y_test, xgb_pred))
    xgb_auc_tuned = float(roc_auc_score(inputs.y_test, xgb_test))

    use_ensemble = best_val_f2 >= (xgb_val_ref + min_f2_gain)
    if use_ensemble:
        selected_model = "ensemble"
        selected_pred = ensemble_pred
        selected_threshold = best_threshold
        selected_f2 = ensemble_f2
        selected_precision = ensemble_precision
        selected_recall = ensemble_recall
        selected_auc = ensemble_auc
    else:
        selected_model = "xgboost"
        selected_pred = xgb_pred
        selected_threshold = xgb_t_best
        selected_f2 = xgb_f2_test_tuned
        selected_precision = xgb_precision_tuned
        selected_recall = xgb_recall_tuned
        selected_auc = xgb_auc_tuned

    tn, fp, fn, tp = confusion_matrix(inputs.y_test, selected_pred).ravel()

    payload: dict[str, Any] = {
        "model": "Weighted Ensemble (validation-tuned)",
        "calibration": "platt_optional",
        "xgb_probability_variant": xgb_variant,
        "lstm_probability_variant": lstm_variant,
        "weights": {"xgboost": best_weight, "lstm": 1.0 - best_weight},
        "best_threshold_val": best_threshold,
        "val_f2_ensemble_best": best_val_f2,
        "xgb_best_threshold_val": xgb_t_best,
        "xgb_best_f2_val_current_split": xgb_f2_best,
        "xgb_val_f2_reference": xgb_val_ref,
        "min_f2_gain_for_ensemble": min_f2_gain,
        "selected_model": selected_model,
        "selected_threshold": selected_threshold,
        "test_f2": selected_f2,
        "test_precision": selected_precision,
        "test_recall": selected_recall,
        "test_roc_auc": selected_auc,
        "ensemble_test_f2": ensemble_f2,
        "ensemble_test_precision": ensemble_precision,
        "ensemble_test_recall": ensemble_recall,
        "ensemble_test_roc_auc": ensemble_auc,
        "xgboost_test_metrics_from_training": {
            "f2_score": float(xgb_metrics_all["test_metrics"]["f2_score"]),
            "precision": float(xgb_metrics_all["test_metrics"]["precision"]),
            "recall": float(xgb_metrics_all["test_metrics"]["recall"]),
            "roc_auc": float(xgb_metrics_all["test_metrics"]["roc_auc"]),
        },
        "lstm_test_metrics_from_training": {
            "f2_score": float(lstm_test_metrics["f2_score"]),
            "precision": float(lstm_test_metrics["precision"]),
            "recall": float(lstm_test_metrics["recall"]),
            "roc_auc": float(lstm_test_metrics["roc_auc"]),
        },
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run validation-tuned ensemble pipeline.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Path to data directory")
    parser.add_argument(
        "--min-f2-gain",
        type=float,
        default=0.005,
        help="Minimum validation F2 gain required to select ensemble",
    )
    parser.add_argument(
        "--no-calibration",
        action="store_true",
        help="Disable Platt calibration when evaluating blend inputs",
    )
    args = parser.parse_args()

    payload = run_ensemble_pipeline(
        data_dir=args.data_dir,
        min_f2_gain=args.min_f2_gain,
        allow_calibration=not args.no_calibration,
    )

    out_path = args.data_dir / "models" / "ensemble_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved: {out_path}")
    print(f"Selected model: {payload['selected_model']}")
    print(f"Selected test F2: {payload['test_f2']:.4f}")
    print(f"Best ensemble val F2: {payload['val_f2_ensemble_best']:.4f}")


if __name__ == "__main__":
    main()
