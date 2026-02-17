"""Streamlit dashboard for model metrics, policy state, and live API checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
import streamlit as st
from fastapi.testclient import TestClient

from src.serving.api import create_app


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = DATA_DIR / "models"
PROCESSED_DIR = DATA_DIR / "processed"


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _metrics_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    xgb = _load_json(MODELS_DIR / "xgboost_baseline_metrics.json")
    lstm = _load_json(MODELS_DIR / "lstm_temporal_metrics.json")
    ens = _load_json(MODELS_DIR / "ensemble_metrics.json")

    if xgb:
        t = xgb.get("test_metrics", {})
        rows.append(
            {
                "model": "xgboost",
                "f2": t.get("f2_score"),
                "precision": t.get("precision"),
                "recall": t.get("recall"),
                "roc_auc": t.get("roc_auc"),
            }
        )
    if lstm:
        t = lstm.get("test_metrics", {})
        rows.append(
            {
                "model": "lstm",
                "f2": t.get("f2_score"),
                "precision": t.get("precision"),
                "recall": t.get("recall"),
                "roc_auc": t.get("roc_auc"),
            }
        )
    if ens:
        rows.append(
            {
                "model": "ensemble_selected",
                "f2": ens.get("test_f2"),
                "precision": ens.get("test_precision"),
                "recall": ens.get("test_recall"),
                "roc_auc": ens.get("test_roc_auc"),
            }
        )

    return pd.DataFrame(rows)


def _build_sequence_from_csv(csv_path: Path, start_row: int, seq_len: int, features: list[str]) -> list[dict[str, float]]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"No rows found in {csv_path}")

    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Input CSV missing required features. Example: {missing[:5]}")

    start = max(0, min(start_row, max(0, len(df) - 1)))
    window = df.iloc[start : start + seq_len][features].copy()
    if window.empty:
        window = df.iloc[:1][features].copy()

    while len(window) < seq_len:
        window.loc[len(window)] = window.iloc[-1]

    records = window.to_dict(orient="records")
    return [{k: float(v) for k, v in row.items()} for row in records]


def _call_api_external(base_url: str, endpoint: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any] | str]:
    url = f"{base_url.rstrip('/')}{endpoint}"
    with httpx.Client(timeout=60.0) as client:
        r = client.post(url, json=payload)
        try:
            return r.status_code, r.json()
        except Exception:
            return r.status_code, r.text


def _call_api_local(endpoint: str, payload: dict[str, Any], data_dir: Path) -> tuple[int, dict[str, Any] | str]:
    app = create_app(data_dir=data_dir)
    with TestClient(app) as client:
        if endpoint.startswith("/explain"):
            path, _, query = endpoint.partition("?")
            r = client.post(f"{path}?{query}" if query else path, json=payload)
        else:
            r = client.post(endpoint, json=payload)
    try:
        return r.status_code, r.json()
    except Exception:
        return r.status_code, r.text


def _render_overview() -> None:
    st.subheader("Model Benchmarks")
    metrics = _metrics_table()
    if metrics.empty:
        st.warning("Metrics artifacts not found in data/models.")
        return

    st.dataframe(metrics, use_container_width=True)
    st.bar_chart(metrics.set_index("model")[["f2", "precision", "recall"]], use_container_width=True)

    ens = _load_json(MODELS_DIR / "ensemble_metrics.json")
    if ens:
        st.subheader("Serving Decision")
        c1, c2, c3 = st.columns(3)
        c1.metric("Selected Model", str(ens.get("selected_model", "unknown")))
        c2.metric("Selected Threshold", f"{float(ens.get('selected_threshold', 0.5)):.3f}")
        c3.metric("Selected Test F2", f"{float(ens.get('test_f2', 0.0)):.4f}")


def _render_ops_status() -> None:
    st.subheader("Ops Status")
    drift = _load_json(MODELS_DIR / "drift_report.json")
    retrain = _load_json(MODELS_DIR / "retraining_decision.json")
    e2e = _load_json(MODELS_DIR / "e2e_real_check_report.json")

    c1, c2, c3 = st.columns(3)
    if drift:
        s = drift.get("summary", {})
        c1.metric("Drift Status", str(s.get("status", "unknown")))
    else:
        c1.metric("Drift Status", "missing")

    if retrain:
        c2.metric("Retraining Triggered", str(retrain.get("retraining_triggered", False)))
        c3.metric("Retraining Reason", str(retrain.get("reason", "n/a")))
    else:
        c2.metric("Retraining Triggered", "missing")
        c3.metric("Retraining Reason", "missing")

    if e2e:
        st.caption(f"Last E2E report: {MODELS_DIR / 'e2e_real_check_report.json'}")
        st.json(e2e, expanded=False)


def _render_live_inference() -> None:
    st.subheader("Live API Inference")

    feature_payload = _load_json(MODELS_DIR / "feature_names.json")
    lstm_payload = _load_json(MODELS_DIR / "lstm_features.json")
    if not feature_payload or not lstm_payload:
        st.info("Need feature_names.json and lstm_features.json in data/models to build sequence payloads.")
        return

    features = [str(x) for x in feature_payload.get("features", [])]
    seq_len = int(lstm_payload.get("sequence_length", 30))
    default_csv = PROCESSED_DIR / "train_features_FD001.csv"

    mode = st.radio(
        "Inference Mode",
        options=["Local In-Process (No API server needed)", "External FastAPI URL"],
        horizontal=True,
    )
    use_local = mode.startswith("Local")
    api_base = st.text_input("API Base URL", value="http://localhost:8000", disabled=use_local)
    csv_path_str = st.text_input("Sequence Source CSV", value=str(default_csv))
    start_row = st.number_input("Start Row", min_value=0, max_value=1_000_000, value=0, step=1)
    equipment_id = st.text_input("Equipment ID", value="streamlit_engine_001")
    top_k = st.slider("Explain Top-K", min_value=1, max_value=20, value=5)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if st.button("Check /health", use_container_width=True):
            try:
                if use_local:
                    app = create_app(data_dir=DATA_DIR)
                    with TestClient(app) as client:
                        r = client.get("/health")
                else:
                    with httpx.Client(timeout=20.0) as client:
                        r = client.get(f"{api_base.rstrip('/')}/health")
                st.write({"status_code": r.status_code, "body": r.json()})
            except Exception as exc:
                st.error(f"Health check failed: {exc}")

    with col_b:
        if st.button("Call /predict", use_container_width=True):
            try:
                payload = {
                    "equipment_id": equipment_id,
                    "sequence": _build_sequence_from_csv(Path(csv_path_str), int(start_row), seq_len, features),
                }
                if use_local:
                    code, body = _call_api_local("/predict", payload, data_dir=DATA_DIR)
                else:
                    code, body = _call_api_external(api_base, "/predict", payload)
                st.write({"status_code": code, "body": body})
            except Exception as exc:
                st.error(f"/predict failed: {exc}")

    with col_c:
        if st.button("Call /explain", use_container_width=True):
            try:
                payload = {
                    "equipment_id": equipment_id,
                    "sequence": _build_sequence_from_csv(Path(csv_path_str), int(start_row), seq_len, features),
                }
                endpoint = f"/explain?top_k={top_k}"
                if use_local:
                    code, body = _call_api_local(endpoint, payload, data_dir=DATA_DIR)
                else:
                    code, body = _call_api_external(api_base, endpoint, payload)
                st.write({"status_code": code, "body": body})
            except Exception as exc:
                st.error(f"/explain failed: {exc}")


def main() -> None:
    st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
    st.title("Predictive Maintenance Dashboard")
    st.caption("Metrics, policy status, drift/retraining state, and live API checks.")

    tab1, tab2, tab3 = st.tabs(["Overview", "Ops", "Live Inference"])
    with tab1:
        _render_overview()
    with tab2:
        _render_ops_status()
    with tab3:
        _render_live_inference()


if __name__ == "__main__":
    main()
