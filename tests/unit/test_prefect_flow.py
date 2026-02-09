"""Unit tests for pipeline orchestration module."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from src.pipelines import prefect_flow


def test_build_parser_defaults() -> None:
    parser = prefect_flow.build_parser()
    args = parser.parse_args([])
    assert args.engine == "local"
    assert args.data_dir == "data"
    assert args.epochs == 100


def test_training_flow_local_with_monkeypatched_steps(monkeypatch) -> None:
    monkeypatch.setattr(prefect_flow, "_run_xgboost_step", lambda **kwargs: {"test_f2": 0.9})  # type: ignore[no-untyped-def]
    monkeypatch.setattr(prefect_flow, "_run_lstm_step", lambda **kwargs: {"test_f2": 0.8})  # type: ignore[no-untyped-def]
    monkeypatch.setattr(  # type: ignore[no-untyped-def]
        prefect_flow, "_run_ensemble_step", lambda **kwargs: {"test_f2": 0.85, "selected_model": "xgboost"}
    )
    out = prefect_flow.training_flow_local(
        data_dir="data",
        sequence_length=30,
        top_k_features=40,
        epochs=2,
        batch_size=64,
        min_f2_gain=0.005,
        no_calibration=False,
        no_mlflow=True,
    )
    assert out["xgboost"]["test_f2"] == 0.9
    assert out["lstm"]["test_f2"] == 0.8
    assert out["ensemble"]["selected_model"] == "xgboost"


def test_save_summary_writes_file(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "models").mkdir(parents=True, exist_ok=True)
    results = {
        "xgboost": {"test_f2": 0.9},
        "lstm": {"test_f2": 0.8},
        "ensemble": {"selected_model": "xgboost", "test_f2": 0.85},
    }
    out = prefect_flow._save_summary(data_dir, results, mode="local", started_at=0.0)
    assert out.exists()
    payload = out.read_text(encoding="utf-8")
    assert "selected_model" in payload


def test_step_wrappers_forward_arguments_and_write_outputs(tmp_path: Path, monkeypatch) -> None:
    data_dir = tmp_path / "data"
    seen: dict[str, object] = {}

    def fake_xgb(**kwargs):  # type: ignore[no-untyped-def]
        seen["xgb"] = kwargs
        return {"test_f2": 0.91}

    def fake_lstm(**kwargs):  # type: ignore[no-untyped-def]
        seen["lstm"] = kwargs
        return {"test_f2": 0.82}

    def fake_ens(**kwargs):  # type: ignore[no-untyped-def]
        seen["ens"] = kwargs
        return {"test_f2": 0.86, "selected_model": "xgboost"}

    monkeypatch.setattr(prefect_flow, "run_xgboost_pipeline", fake_xgb)
    monkeypatch.setattr(prefect_flow, "run_lstm_pipeline", fake_lstm)
    monkeypatch.setattr(prefect_flow, "run_ensemble_pipeline", fake_ens)

    xgb = prefect_flow._run_xgboost_step(
        data_dir=data_dir,
        enable_mlflow=False,
        register_model=True,
        registered_model_name="xgb-name",
    )
    lstm = prefect_flow._run_lstm_step(
        data_dir=data_dir,
        sequence_length=30,
        top_k_features=40,
        epochs=2,
        batch_size=64,
        enable_mlflow=False,
        register_model=True,
        registered_model_name="lstm-name",
    )
    ens = prefect_flow._run_ensemble_step(
        data_dir=data_dir,
        min_f2_gain=0.01,
        allow_calibration=False,
        enable_mlflow=False,
    )

    assert xgb["test_f2"] == 0.91
    assert lstm["test_f2"] == 0.82
    assert ens["selected_model"] == "xgboost"
    assert seen["xgb"] == {
        "data_dir": data_dir,
        "enable_mlflow": False,
        "register_model": True,
        "registered_model_name": "xgb-name",
    }
    assert seen["lstm"] == {
        "data_dir": data_dir,
        "sequence_length": 30,
        "top_k_features": 40,
        "epochs": 2,
        "batch_size": 64,
        "enable_mlflow": False,
        "register_model": True,
        "registered_model_name": "lstm-name",
    }
    assert seen["ens"] == {
        "data_dir": data_dir,
        "min_f2_gain": 0.01,
        "allow_calibration": False,
        "enable_mlflow": False,
    }
    assert (data_dir / "models" / "ensemble_metrics.json").exists()


def test_main_local_engine_emits_summary_and_stdout(tmp_path: Path, monkeypatch, capsys) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "models").mkdir(parents=True, exist_ok=True)
    fake_results = {
        "xgboost": {"test_f2": 0.90},
        "lstm": {"test_f2": 0.80},
        "ensemble": {"selected_model": "xgboost", "test_f2": 0.85},
    }

    monkeypatch.setattr(
        prefect_flow,
        "training_flow_local",
        lambda **kwargs: fake_results,  # type: ignore[no-untyped-def]
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["prog", "--engine", "local", "--data-dir", str(data_dir), "--no-mlflow"],
    )

    prefect_flow.main()
    out = capsys.readouterr().out
    assert "Pipeline orchestration complete." in out
    assert (data_dir / "models" / "pipeline_run_summary.json").exists()


def test_main_prefect_engine_raises_when_prefect_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(prefect_flow, "PREFECT_AVAILABLE", False)
    monkeypatch.setattr(sys, "argv", ["prog", "--engine", "prefect"])
    with pytest.raises(RuntimeError, match="Prefect is not installed"):
        prefect_flow.main()


def test_reload_with_fake_prefect_and_run_prefect_flow(monkeypatch) -> None:
    class FakeLogger:
        def info(self, *args, **kwargs):  # type: ignore[no-untyped-def]
            return None

    class FakePrefect:
        @staticmethod
        def flow(*args, **kwargs):  # type: ignore[no-untyped-def]
            def deco(fn):
                return fn

            return deco

        @staticmethod
        def task(*args, **kwargs):  # type: ignore[no-untyped-def]
            def deco(fn):
                return fn

            return deco

        @staticmethod
        def get_run_logger():  # type: ignore[no-untyped-def]
            return FakeLogger()

    original_prefect = sys.modules.get("prefect")
    monkeypatch.setitem(sys.modules, "prefect", FakePrefect)
    module = importlib.reload(prefect_flow)

    monkeypatch.setattr(module, "_run_xgboost_step", lambda **kwargs: {"test_f2": 0.9})  # type: ignore[no-untyped-def]
    monkeypatch.setattr(module, "_run_lstm_step", lambda **kwargs: {"test_f2": 0.8})  # type: ignore[no-untyped-def]
    monkeypatch.setattr(module, "train_ensemble_task", lambda *args, **kwargs: {"test_f2": 0.85, "selected_model": "xgboost"})  # type: ignore[no-untyped-def]

    out = module.training_flow_prefect(  # type: ignore[attr-defined]
        data_dir="data",
        no_mlflow=True,
        register_models=True,
        xgb_registered_model_name="xgb-rm",
        lstm_registered_model_name="lstm-rm",
    )
    assert out["ensemble"]["selected_model"] == "xgboost"
    assert module.PREFECT_AVAILABLE is True
    assert module._logger().__class__.__name__ == "FakeLogger"

    if original_prefect is None:
        sys.modules.pop("prefect", None)
    else:
        sys.modules["prefect"] = original_prefect
    importlib.reload(module)
