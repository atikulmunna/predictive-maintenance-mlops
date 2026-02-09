"""Drift detection for engineered features and model predictions.

This module compares a reference dataset (typically training/validation window)
to a current dataset (latest inference or recent batch) and writes a drift
report to ``data/models/drift_report.json`` by default.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def _safe_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan).dropna()


def _population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    if reference.size == 0 or current.size == 0:
        return 0.0

    # Quantile bins on reference preserve expected distribution mass.
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size <= 2:
        return 0.0

    reference_hist, _ = np.histogram(reference, bins=edges)
    current_hist, _ = np.histogram(current, bins=edges)

    ref_ratio = reference_hist / max(reference_hist.sum(), 1)
    cur_ratio = current_hist / max(current_hist.sum(), 1)

    eps = 1e-8
    ref_ratio = np.clip(ref_ratio, eps, 1.0)
    cur_ratio = np.clip(cur_ratio, eps, 1.0)
    psi = np.sum((cur_ratio - ref_ratio) * np.log(cur_ratio / ref_ratio))
    return float(psi)


def detect_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list[str],
    *,
    p_value_threshold: float = 0.01,
    psi_threshold: float = 0.2,
) -> dict[str, Any]:
    features: list[dict[str, Any]] = []
    drifted_count = 0

    for feature in feature_columns:
        if feature not in reference_df.columns or feature not in current_df.columns:
            continue
        ref = _safe_numeric(reference_df[feature])
        cur = _safe_numeric(current_df[feature])
        if ref.empty or cur.empty:
            continue

        ks_stat, p_value = ks_2samp(ref.to_numpy(), cur.to_numpy())
        psi = _population_stability_index(ref.to_numpy(), cur.to_numpy())
        is_drift = (p_value < p_value_threshold) or (psi >= psi_threshold)
        if is_drift:
            drifted_count += 1

        features.append(
            {
                "feature": feature,
                "ks_statistic": float(ks_stat),
                "ks_p_value": float(p_value),
                "psi": psi,
                "is_drift": bool(is_drift),
            }
        )

    return {
        "feature_count_checked": len(features),
        "drifted_feature_count": drifted_count,
        "features": features,
        "thresholds": {"ks_p_value": p_value_threshold, "psi": psi_threshold},
    }


def detect_prediction_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    prediction_column: str,
    *,
    p_value_threshold: float = 0.01,
    mean_shift_threshold: float = 0.05,
) -> dict[str, Any] | None:
    if prediction_column not in reference_df.columns or prediction_column not in current_df.columns:
        return None

    ref = _safe_numeric(reference_df[prediction_column])
    cur = _safe_numeric(current_df[prediction_column])
    if ref.empty or cur.empty:
        return None

    ks_stat, p_value = ks_2samp(ref.to_numpy(), cur.to_numpy())
    mean_shift = abs(float(cur.mean() - ref.mean()))
    is_drift = (p_value < p_value_threshold) or (mean_shift >= mean_shift_threshold)
    return {
        "prediction_column": prediction_column,
        "ks_statistic": float(ks_stat),
        "ks_p_value": float(p_value),
        "mean_reference": float(ref.mean()),
        "mean_current": float(cur.mean()),
        "mean_shift_abs": mean_shift,
        "thresholds": {"ks_p_value": p_value_threshold, "mean_shift_abs": mean_shift_threshold},
        "is_drift": bool(is_drift),
    }


def load_feature_columns(data_dir: Path) -> list[str]:
    feature_path = data_dir / "models" / "feature_names.json"
    if not feature_path.exists():
        return []
    payload = json.loads(feature_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and isinstance(payload.get("features"), list):
        return [str(c) for c in payload["features"]]
    return []


def run_drift_detection(
    reference_path: Path,
    current_path: Path,
    *,
    data_dir: Path = Path("data"),
    feature_columns: list[str] | None = None,
    prediction_column: str = "failure_probability",
    feature_p_value_threshold: float = 0.01,
    feature_psi_threshold: float = 0.2,
    min_drifted_features: int = 3,
    prediction_mean_shift_threshold: float = 0.05,
    output_path: Path | None = None,
) -> dict[str, Any]:
    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    columns = feature_columns if feature_columns is not None else load_feature_columns(data_dir)
    if not columns:
        ignored = {"label", "target", "y", "failure_probability"}
        columns = [c for c in reference_df.columns if c in current_df.columns and c not in ignored]

    feature_payload = detect_feature_drift(
        reference_df=reference_df,
        current_df=current_df,
        feature_columns=columns,
        p_value_threshold=feature_p_value_threshold,
        psi_threshold=feature_psi_threshold,
    )
    pred_payload = detect_prediction_drift(
        reference_df=reference_df,
        current_df=current_df,
        prediction_column=prediction_column,
        p_value_threshold=feature_p_value_threshold,
        mean_shift_threshold=prediction_mean_shift_threshold,
    )

    drift_detected = feature_payload["drifted_feature_count"] >= min_drifted_features
    if pred_payload is not None:
        drift_detected = drift_detected or pred_payload["is_drift"]

    summary = {
        "status": "drift_detected" if drift_detected else "stable",
        "drift_detected": bool(drift_detected),
        "min_drifted_features": int(min_drifted_features),
        "feature_drifted_count": int(feature_payload["drifted_feature_count"]),
        "prediction_drift_detected": bool(pred_payload["is_drift"]) if pred_payload is not None else False,
    }
    payload = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reference_path": str(reference_path),
        "current_path": str(current_path),
        "summary": summary,
        "feature_drift": feature_payload,
        "prediction_drift": pred_payload,
    }

    out_path = output_path or (data_dir / "models" / "drift_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["output_path"] = str(out_path)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Detect feature/prediction drift between two CSV datasets.")
    parser.add_argument("--reference", required=True, type=Path, help="Reference CSV path")
    parser.add_argument("--current", required=True, type=Path, help="Current CSV path")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Project data directory")
    parser.add_argument(
        "--prediction-column",
        default="failure_probability",
        help="Prediction probability column to evaluate for drift",
    )
    parser.add_argument("--feature-p-threshold", type=float, default=0.01, help="KS p-value threshold")
    parser.add_argument("--feature-psi-threshold", type=float, default=0.2, help="PSI threshold")
    parser.add_argument("--prediction-mean-shift-threshold", type=float, default=0.05, help="Prediction mean shift")
    parser.add_argument(
        "--min-drifted-features",
        type=int,
        default=3,
        help="Minimum drifted feature count required to mark feature drift",
    )
    parser.add_argument("--output", type=Path, default=None, help="Output report path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = run_drift_detection(
        reference_path=args.reference,
        current_path=args.current,
        data_dir=args.data_dir,
        prediction_column=args.prediction_column,
        feature_p_value_threshold=args.feature_p_threshold,
        feature_psi_threshold=args.feature_psi_threshold,
        min_drifted_features=args.min_drifted_features,
        prediction_mean_shift_threshold=args.prediction_mean_shift_threshold,
        output_path=args.output,
    )
    summary = payload["summary"]
    print("Drift detection complete.")
    print(f"- status: {summary['status']}")
    print(f"- feature drifted count: {summary['feature_drifted_count']}")
    print(f"- prediction drift: {summary['prediction_drift_detected']}")
    print(f"- report: {payload['output_path']}")


if __name__ == "__main__":
    main()
