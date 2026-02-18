"""Ensure CI has a minimal processed training dataset.

Creates data/processed/train_features_FD001.csv when missing.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    out_path = Path("data/processed/train_features_FD001.csv")
    if out_path.exists():
        print(f"Processed dataset already exists: {out_path}")
        return

    rng = np.random.default_rng(42)
    n_engines = 30
    cycles_per_engine = 50
    n_features = 45
    rows = []

    for unit_id in range(1, n_engines + 1):
        for cycle in range(1, cycles_per_engine + 1):
            rul = cycles_per_engine - cycle
            failure_soon = 1 if rul <= 10 else 0
            signal = (1.0 / (rul + 1.0)) + float(rng.normal(0.0, 0.02))
            row = {
                "unit_id": unit_id,
                "cycle": cycle,
                "RUL": rul,
                "failure_soon": failure_soon,
                "f_signal": signal,
                "f_cycle_norm": cycle / cycles_per_engine,
            }
            for i in range(1, n_features + 1):
                trend = 0.0 if i % 3 else cycle * 0.005
                row[f"f_{i:02d}"] = float(rng.normal(0.0, 1.0) + trend)
            rows.append(row)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Created synthetic dataset: {out_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
