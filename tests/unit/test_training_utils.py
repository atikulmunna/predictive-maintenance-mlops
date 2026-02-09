"""Unit tests for training utility modules."""

from __future__ import annotations

import builtins
from pathlib import Path

from src.training import mlflow_utils
from src.training import trainer
from src.training.trainer import build_parser


def test_build_parser_accepts_no_mlflow_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["--model", "ensemble", "--no-mlflow"])
    assert args.model == "ensemble"
    assert args.no_mlflow is True


def test_get_mlflow_client_import_failure_returns_none(monkeypatch) -> None:
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "mlflow":
            raise ImportError("mlflow unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert mlflow_utils.get_mlflow_client(Path("data"), "exp") is None


def test_get_mlflow_client_success_and_log_artifacts(tmp_path, monkeypatch) -> None:
    class DummyMlflow:
        tracking_uri = None
        experiment = None
        artifacts: list[str] = []

        @staticmethod
        def set_tracking_uri(uri: str) -> None:
            DummyMlflow.tracking_uri = uri

        @staticmethod
        def set_experiment(name: str) -> None:
            DummyMlflow.experiment = name

        @staticmethod
        def log_artifact(path: str) -> None:
            DummyMlflow.artifacts.append(path)

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "mlflow":
            return DummyMlflow
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    client = mlflow_utils.get_mlflow_client(Path("data"), "demo-exp")
    assert client is DummyMlflow
    assert DummyMlflow.experiment == "demo-exp"
    assert DummyMlflow.tracking_uri is not None

    existing = tmp_path / "exists.txt"
    missing = tmp_path / "missing.txt"
    existing.write_text("ok", encoding="utf-8")
    mlflow_utils.log_artifacts_if_exist(DummyMlflow, [existing, missing])
    assert str(existing) in DummyMlflow.artifacts
    assert str(missing) not in DummyMlflow.artifacts


def test_register_model_if_possible_paths() -> None:
    skipped = mlflow_utils.register_model_if_possible(None, "abc", "model", "name")
    assert skipped["status"] == "skipped"

    skipped2 = mlflow_utils.register_model_if_possible(object(), None, "model", "name")
    assert skipped2["status"] == "skipped"

    class DummyMlflow:
        @staticmethod
        def register_model(model_uri: str, name: str):  # type: ignore[no-untyped-def]
            class V:
                version = 7

            assert model_uri.startswith("runs:/")
            assert name == "demo-model"
            return V()

    ok = mlflow_utils.register_model_if_possible(DummyMlflow, "run123", "model", "demo-model")
    assert ok["status"] == "registered"
    assert ok["version"] == "7"


def test_write_registry_state(tmp_path: Path) -> None:
    out = mlflow_utils.write_registry_state(tmp_path, {"selected_model": "xgboost"})
    assert out.exists()
    assert "selected_model" in out.read_text(encoding="utf-8")


def test_trainer_run_selected_pipeline_all_branches(monkeypatch) -> None:
    calls: list[str] = []

    def fake_xgb(**kwargs):  # type: ignore[no-untyped-def]
        calls.append("xgb")
        return {"test_f2": 0.9}

    def fake_lstm(**kwargs):  # type: ignore[no-untyped-def]
        calls.append("lstm")
        return {"test_f2": 0.8}

    def fake_ens(**kwargs):  # type: ignore[no-untyped-def]
        calls.append("ens")
        return {"test_f2": 0.85, "selected_model": "xgboost"}

    monkeypatch.setattr(trainer, "run_xgboost_pipeline", fake_xgb)
    monkeypatch.setattr(trainer, "run_lstm_pipeline", fake_lstm)
    monkeypatch.setattr(trainer, "run_ensemble_pipeline", fake_ens)

    args = build_parser().parse_args(["--model", "all", "--data-dir", "data", "--no-mlflow"])
    result = trainer.run_selected_pipeline(args)
    assert set(result.keys()) == {"xgboost", "lstm", "ensemble"}
    assert calls == ["xgb", "lstm", "ens"]


def test_trainer_single_model_paths(monkeypatch) -> None:
    monkeypatch.setattr(trainer, "run_xgboost_pipeline", lambda **kwargs: {"test_f2": 0.9})  # type: ignore[no-untyped-def]
    monkeypatch.setattr(trainer, "run_lstm_pipeline", lambda **kwargs: {"test_f2": 0.8})  # type: ignore[no-untyped-def]
    monkeypatch.setattr(  # type: ignore[no-untyped-def]
        trainer,
        "run_ensemble_pipeline",
        lambda **kwargs: {"test_f2": 0.85, "selected_model": "xgboost"},
    )

    args_xgb = build_parser().parse_args(["--model", "xgboost", "--data-dir", "data", "--no-mlflow"])
    assert "xgboost" in trainer.run_selected_pipeline(args_xgb)

    args_lstm = build_parser().parse_args(["--model", "lstm", "--data-dir", "data", "--no-mlflow"])
    assert "lstm" in trainer.run_selected_pipeline(args_lstm)

    args_ens = build_parser().parse_args(["--model", "ensemble", "--data-dir", "data", "--no-mlflow"])
    assert "ensemble" in trainer.run_selected_pipeline(args_ens)


def test_trainer_all_writes_registry_state(monkeypatch, tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "models").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(trainer, "run_xgboost_pipeline", lambda **kwargs: {"test_f2": 0.9, "mlflow_run_id": "r1", "mlflow_registry": {"status": "skipped"}})  # type: ignore[no-untyped-def]
    monkeypatch.setattr(trainer, "run_lstm_pipeline", lambda **kwargs: {"test_f2": 0.8, "mlflow_run_id": "r2", "mlflow_registry": {"status": "skipped"}})  # type: ignore[no-untyped-def]
    monkeypatch.setattr(  # type: ignore[no-untyped-def]
        trainer,
        "run_ensemble_pipeline",
        lambda **kwargs: {"test_f2": 0.85, "selected_model": "xgboost", "selected_threshold": 0.5},
    )
    args = build_parser().parse_args(["--model", "all", "--data-dir", str(data_dir), "--no-mlflow"])
    out = trainer.run_selected_pipeline(args)
    assert out["ensemble"]["selected_model"] == "xgboost"
    assert (data_dir / "models" / "model_registry_state.json").exists()
