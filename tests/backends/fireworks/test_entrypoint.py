from unittest.mock import AsyncMock, MagicMock

import pytest

from skyrl.backends.fireworks.runtime import FireworksRuntime
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.entrypoints.main_base import BasePPOExp
from skyrl.train.entrypoints.main_fireworks import (
    FullyAsyncFireworksExp,
    experiment_class,
)


def test_fireworks_provider_preflight_runs_before_tracker(monkeypatch, tmp_path) -> None:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "fireworks"
    cfg.trainer.fireworks.base_model = "accounts/fireworks/models/test"
    cfg.trainer.fireworks.training_shape_id = "accounts/fireworks/trainingShapes/test"
    cfg.trainer.fireworks.trainer_job_id = "skyrl-smoke-test-trainer"
    cfg.trainer.fireworks.deployment_id = "skyrl-smoke-test-rollout"
    cfg.trainer.policy.model.lora.rank = 8
    cfg.trainer.export_path = str(tmp_path / "exports")
    cfg.trainer.ckpt_path = str(tmp_path / "checkpoints")

    exp = object.__new__(BasePPOExp)
    exp.cfg = cfg
    exp.tokenizer = object()
    exp._fireworks_runtime = None

    tracker_started = False

    def _start_tracker():
        nonlocal tracker_started
        tracker_started = True
        raise AssertionError("tracker must not start before provider preflight")

    def _reject_provider(**_kwargs):
        raise RuntimeError("dedicated trainer provisioning failed")

    monkeypatch.setattr(exp, "get_tracker", _start_tracker)
    monkeypatch.setattr(FireworksRuntime, "connect", staticmethod(_reject_provider))

    with pytest.raises(RuntimeError, match="dedicated trainer provisioning"):
        exp._setup_trainer()

    assert not tracker_started


def test_fireworks_runtime_usage_is_registered_with_tracker(
    monkeypatch, tmp_path
) -> None:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "fireworks"
    cfg.trainer.fireworks.base_model = "accounts/fireworks/models/test"
    cfg.trainer.fireworks.training_shape_id = "accounts/fireworks/trainingShapes/test"
    cfg.trainer.fireworks.trainer_job_id = "skyrl-smoke-test-trainer"
    cfg.trainer.fireworks.deployment_id = "skyrl-smoke-test-rollout"
    cfg.trainer.policy.model.lora.rank = 8
    cfg.trainer.export_path = str(tmp_path / "exports")
    cfg.trainer.ckpt_path = str(tmp_path / "checkpoints")

    exp = object.__new__(BasePPOExp)
    exp.cfg = cfg
    exp.tokenizer = object()
    exp.train_dataset = object()
    exp.eval_dataset = object()
    exp._fireworks_runtime = None

    runtime = MagicMock()
    tracker = MagicMock()
    trainer = MagicMock()
    monkeypatch.setattr(FireworksRuntime, "connect", lambda **_kwargs: runtime)
    monkeypatch.setattr(exp, "get_tracker", lambda: tracker)
    monkeypatch.setattr(exp, "get_generator", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(exp, "get_trainer", lambda **_kwargs: trainer)
    monkeypatch.setattr(exp, "get_trajectory_logger", lambda: object())

    exp._setup_trainer()

    tracker.set_metrics_provider.assert_called_once_with(
        runtime.usage_metrics,
        summary_provider=runtime.usage_summary,
    )


def test_direct_entrypoint_selects_fully_async_scheduler() -> None:
    cfg = SkyRLTrainConfig()
    assert experiment_class(cfg) is BasePPOExp

    cfg.trainer.fully_async.enabled = True
    assert experiment_class(cfg) is FullyAsyncFireworksExp


def _run_exp_with_fireworks_runtime(*, train_error: Exception | None):
    exp = object.__new__(BasePPOExp)
    runtime = MagicMock()
    runtime.close = AsyncMock()
    trainer = MagicMock()
    trainer.train = AsyncMock(side_effect=train_error)
    exp._fireworks_runtime = runtime
    exp._tinker_runtime = None
    exp._setup_trainer = MagicMock(return_value=trainer)
    return exp, runtime


def test_failed_training_preserves_fireworks_trainer_record() -> None:
    exp, runtime = _run_exp_with_fireworks_runtime(
        train_error=RuntimeError("training failed")
    )

    with pytest.raises(RuntimeError, match="training failed"):
        exp.run()

    runtime.close.assert_awaited_once_with(preserve_trainer=True)


def test_successful_training_uses_normal_fireworks_cleanup() -> None:
    exp, runtime = _run_exp_with_fireworks_runtime(train_error=None)

    exp.run()

    runtime.close.assert_awaited_once_with(preserve_trainer=False)
