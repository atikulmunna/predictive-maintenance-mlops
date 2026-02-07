"""Train/evaluate/save the temporal LSTM model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import fbeta_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks, layers, models
from tensorflow.keras.optimizers import Adam

from src.training.mlflow_utils import get_mlflow_client, log_artifacts_if_exist


def create_sequences(
    df: pd.DataFrame, feature_cols: list[str], sequence_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_sequences: list[np.ndarray] = []
    y_sequences: list[float] = []
    engine_ids: list[Any] = []
    for engine_id in df["unit_id"].unique():
        engine_data = df[df["unit_id"] == engine_id].sort_values("cycle")
        features = engine_data[feature_cols].values
        targets = engine_data["failure_soon"].values
        for i in range(len(features) - sequence_length + 1):
            x_sequences.append(features[i : i + sequence_length])
            y_sequences.append(targets[i + sequence_length - 1])
            engine_ids.append(engine_id)
    return np.array(x_sequences), np.array(y_sequences), np.array(engine_ids)


def split_by_engine(engine_ids_seq: np.ndarray, engines: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_engines = len(engines)
    train_engines = set(engines[: int(0.7 * n_engines)])
    val_engines = set(engines[int(0.7 * n_engines) : int(0.85 * n_engines)])
    test_engines = set(engines[int(0.85 * n_engines) :])
    train_mask = np.array([eid in train_engines for eid in engine_ids_seq])
    val_mask = np.array([eid in val_engines for eid in engine_ids_seq])
    test_mask = np.array([eid in test_engines for eid in engine_ids_seq])
    return train_mask, val_mask, test_mask


def create_lstm_model(seq_length: int, n_features: int, attention: bool = True) -> models.Model:
    inputs = layers.Input(shape=(seq_length, n_features))
    x = layers.LSTM(64, return_sequences=True, activation="relu")(inputs)
    x = layers.Dropout(0.2)(x)
    x = layers.LSTM(32, return_sequences=attention, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    if attention:
        x = layers.Attention()([x, x])
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC()],
    )
    return model


def run_lstm_pipeline(
    data_dir: Path,
    sequence_length: int = 30,
    top_k_features: int = 40,
    epochs: int = 100,
    batch_size: int = 32,
    enable_mlflow: bool = True,
    mlflow_experiment: str = "turbofan_lstm_temporal",
) -> dict[str, Any]:
    processed_path = data_dir / "processed" / "train_features_FD001.csv"
    models_dir = data_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(processed_path)
    exclude_cols = ["unit_id", "cycle", "RUL", "failure_soon"]
    all_features = [col for col in df.columns if col not in exclude_cols]
    correlations = df[all_features + ["failure_soon"]].corr()["failure_soon"].drop("failure_soon").abs()
    top_features = correlations.nlargest(top_k_features).index.tolist()

    x_seq, y_seq, engine_ids = create_sequences(df, top_features, sequence_length)
    train_mask, val_mask, test_mask = split_by_engine(engine_ids, df["unit_id"].unique())

    x_train = x_seq[train_mask]
    y_train = y_seq[train_mask]
    x_val = x_seq[val_mask]
    y_val = y_seq[val_mask]
    x_test = x_seq[test_mask]
    y_test = y_seq[test_mask]

    scaler = StandardScaler()
    n_train, n_steps, n_features = x_train.shape
    x_train_scaled = scaler.fit_transform(x_train.reshape(-1, n_features)).reshape(n_train, n_steps, n_features)
    x_val_scaled = scaler.transform(x_val.reshape(-1, n_features)).reshape(x_val.shape)
    x_test_scaled = scaler.transform(x_test.reshape(-1, n_features)).reshape(x_test.shape)

    model = create_lstm_model(sequence_length, len(top_features), attention=True)
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss", patience=10, restore_best_weights=True, verbose=0
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=5, min_lr=0.00001, verbose=0
    )

    history = model.fit(
        x_train_scaled,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_val_scaled, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=0,
    )

    y_val_proba = model.predict(x_val_scaled, verbose=0).flatten()
    y_val_pred = (y_val_proba > 0.5).astype(int)
    y_test_proba = model.predict(x_test_scaled, verbose=0).flatten()
    y_test_pred = (y_test_proba > 0.5).astype(int)

    f2_val = float(fbeta_score(y_val, y_val_pred, beta=2))
    precision_val = float(precision_score(y_val, y_val_pred))
    recall_val = float(recall_score(y_val, y_val_pred))
    roc_auc_val = float(roc_auc_score(y_val, y_val_proba))

    f2_test = float(fbeta_score(y_test, y_test_pred, beta=2))
    precision_test = float(precision_score(y_test, y_test_pred))
    recall_test = float(recall_score(y_test, y_test_pred))
    roc_auc_test = float(roc_auc_score(y_test, y_test_proba))

    model_path = models_dir / "lstm_temporal.h5"
    scaler_path = models_dir / "lstm_scaler.pkl"
    features_path = models_dir / "lstm_features.json"
    metrics_path = models_dir / "lstm_temporal_metrics.json"

    model.save(model_path)
    joblib.dump(scaler, scaler_path)
    with open(features_path, "w", encoding="utf-8") as f:
        json.dump({"features": top_features, "sequence_length": sequence_length}, f, indent=2)

    metrics = {
        "model": "LSTM Temporal",
        "timestamp": pd.Timestamp.now().isoformat(),
        "architecture": {
            "sequence_length": sequence_length,
            "n_features": len(top_features),
            "lstm_units": [64, 32],
            "dropout": 0.2,
            "attention": True,
        },
        "training": {
            "batch_size": batch_size,
            "epochs": len(history.history["loss"]),
            "early_stopping": True,
        },
        "validation_metrics": {
            "f2_score": f2_val,
            "precision": precision_val,
            "recall": recall_val,
            "roc_auc": roc_auc_val,
        },
        "test_metrics": {
            "f2_score": f2_test,
            "precision": precision_test,
            "recall": recall_test,
            "roc_auc": roc_auc_test,
        },
        "dataset_info": {
            "train_sequences": int(len(x_train_scaled)),
            "val_sequences": int(len(x_val_scaled)),
            "test_sequences": int(len(x_test_scaled)),
            "n_features_used": int(len(top_features)),
        },
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    mlflow_run_id = None
    if enable_mlflow:
        mlflow = get_mlflow_client(data_dir=data_dir, experiment_name=mlflow_experiment)
        if mlflow is not None:
            with mlflow.start_run(run_name="lstm_temporal_script") as run:
                mlflow.log_params(
                    {
                        "sequence_length": sequence_length,
                        "top_k_features": top_k_features,
                        "batch_size": batch_size,
                        "epochs_trained": len(history.history["loss"]),
                        "attention": True,
                    }
                )
                mlflow.log_metrics(
                    {
                        "val_f2_score": f2_val,
                        "val_precision": precision_val,
                        "val_recall": recall_val,
                        "val_roc_auc": roc_auc_val,
                        "test_f2_score": f2_test,
                        "test_precision": precision_test,
                        "test_recall": recall_test,
                        "test_roc_auc": roc_auc_test,
                    }
                )
                try:
                    mlflow.tensorflow.log_model(model, artifact_path="model")
                except Exception:
                    log_artifacts_if_exist(mlflow, [model_path])
                log_artifacts_if_exist(mlflow, [features_path, scaler_path, metrics_path])
                mlflow_run_id = run.info.run_id

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "features_path": str(features_path),
        "metrics_path": str(metrics_path),
        "validation_f2": f2_val,
        "test_f2": f2_test,
        "test_precision": precision_test,
        "test_recall": recall_test,
        "test_roc_auc": roc_auc_test,
        "epochs_trained": len(history.history["loss"]),
        "mlflow_run_id": mlflow_run_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and persist LSTM temporal artifacts.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Path to data directory")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--sequence-length", type=int, default=30, help="Sequence length")
    parser.add_argument("--top-k-features", type=int, default=40, help="Number of top correlated features")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow logging")
    args = parser.parse_args()

    result = run_lstm_pipeline(
        data_dir=args.data_dir,
        sequence_length=args.sequence_length,
        top_k_features=args.top_k_features,
        epochs=args.epochs,
        batch_size=args.batch_size,
        enable_mlflow=not args.no_mlflow,
    )
    print(f"Saved model: {result['model_path']}")
    print(f"Saved scaler: {result['scaler_path']}")
    print(f"Saved features: {result['features_path']}")
    print(f"Saved metrics: {result['metrics_path']}")
    print(
        "Test metrics: "
        f"F2={result['test_f2']:.4f}, "
        f"Precision={result['test_precision']:.4f}, "
        f"Recall={result['test_recall']:.4f}, "
        f"ROC-AUC={result['test_roc_auc']:.4f}"
    )
    print(f"Epochs trained: {result['epochs_trained']}")
    if result.get("mlflow_run_id"):
        print(f"MLflow run: {result['mlflow_run_id']}")


if __name__ == "__main__":
    main()
