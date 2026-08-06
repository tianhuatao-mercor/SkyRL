import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.tinker.training_backend import TinkerPolicyDispatch
from skyrl.train.config import SkyRLTrainConfig


class _Future:
    def __init__(self, value):
        self.value = value
        self.timeout = None

    def result(self, timeout=None):
        self.timeout = timeout
        return self.value


class _TrainingClient:
    def __init__(self):
        self.forward_backward_calls = []
        self.optim_params = []
        self.saved_states = []
        self.loaded_states = []

    def forward_backward(self, datums, loss_fn):
        self.forward_backward_calls.append((datums, loss_fn))
        return _Future(
            SimpleNamespace(
                metrics={"loss:sum": 1.25, "response_tokens": 4.0},
                loss_fn_output_type="scalar",
            )
        )

    def optim_step(self, params):
        self.optim_params.append(params)
        return _Future(SimpleNamespace(metrics={"grad_norm": 0.75}))

    def save_state(self, name, ttl_seconds=None):
        self.saved_states.append((name, ttl_seconds))
        return _Future(SimpleNamespace(path=f"tinker://source/weights/{name}"))

    def load_state_with_optimizer(self, path):
        self.loaded_states.append((path, True))
        return _Future(SimpleNamespace())

    def load_state(self, path):
        self.loaded_states.append((path, False))
        return _Future(SimpleNamespace())


def _cfg() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.tinker.max_seq_len = 128
    return cfg


def test_dispatch_stages_and_submits_importance_sampling() -> None:
    training_client = _TrainingClient()
    runtime = SimpleNamespace(
        training_client=training_client,
        record_forward_backward=MagicMock(),
    )
    built_batches = []

    def datum_builder(batch, *, max_seq_len):
        built_batches.append((batch, max_seq_len))
        return ["datum"] * batch.batch_size

    dispatch = TinkerPolicyDispatch(_cfg(), runtime, datum_builder=datum_builder)
    batch = TrainingInputBatch(
        {
            "sequences": torch.arange(12).reshape(4, 3),
            "attention_mask": torch.ones((4, 3), dtype=torch.bool),
        }
    )
    staged = dispatch.stage_data("policy", batch, [(0, 2), (2, 4)])
    output = dispatch.forward_backward_from_staged("policy", staged[0])

    assert [part.batch_size for part in staged] == [2, 2]
    assert training_client.forward_backward_calls == [(["datum", "datum"], "importance_sampling")]
    assert built_batches[0][1] == 128
    assert output.metrics["final_loss"] == pytest.approx(1.25)
    runtime.record_forward_backward.assert_called_once()
    assert runtime.record_forward_backward.call_args.kwargs["training_tokens"] == 6
    assert runtime.record_forward_backward.call_args.kwargs["succeeded"] is True


def test_dispatch_optimizer_uses_skyrl_config() -> None:
    cfg = _cfg()
    cfg.trainer.policy.optimizer_config.lr = 2e-5
    cfg.trainer.policy.optimizer_config.adam_betas = [0.8, 0.9]
    cfg.trainer.policy.optimizer_config.weight_decay = 0.1
    cfg.trainer.policy.optimizer_config.max_grad_norm = 2.0
    training_client = _TrainingClient()
    dispatch = TinkerPolicyDispatch(cfg, SimpleNamespace(training_client=training_client))

    grad_norm = dispatch.optim_step("policy")

    params = training_client.optim_params[0]
    assert params.learning_rate == pytest.approx(2e-5)
    assert params.beta1 == pytest.approx(0.8)
    assert params.beta2 == pytest.approx(0.9)
    assert params.weight_decay == pytest.approx(0.1)
    assert params.grad_clip_norm == pytest.approx(2.0)
    assert grad_norm == pytest.approx(0.75)


def test_dispatch_saves_and_loads_checkpoint(tmp_path) -> None:
    training_client = _TrainingClient()
    cfg = _cfg()
    cfg.trainer.tinker.request_timeout_s = 123
    cfg.trainer.tinker.checkpoint_ttl_seconds = 3600
    runtime = SimpleNamespace(
        training_client=training_client,
        record_checkpoint=MagicMock(),
        usage_report=lambda: {
            "cumulative_across_resumes": True,
            "metrics": {"tinker/usage/checkpoints_total": 1},
        },
        restore_usage_reports=MagicMock(),
    )
    dispatch = TinkerPolicyDispatch(cfg, runtime)
    ckpt_dir = tmp_path / "global_step_7" / "policy"

    dispatch.save_checkpoint("policy", str(ckpt_dir), tokenizer="unused")

    checkpoint_name, ttl = training_client.saved_states[0]
    assert checkpoint_name.startswith("skyrl-step-7-")
    assert ttl == 3600
    manifest = json.loads((ckpt_dir / "tinker_checkpoint.json").read_text())
    assert manifest["provider_path"].startswith("tinker://")
    assert manifest["includes_optimizer_state"] is True
    assert manifest["global_step"] == 7
    assert manifest["usage_at_checkpoint"]["metrics"]["tinker/usage/checkpoints_total"] == 1
    runtime.record_checkpoint.assert_called_once()

    dispatch.load_checkpoint(
        "policy",
        str(ckpt_dir),
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
    )

    assert training_client.loaded_states == [(manifest["provider_path"], True)]
    runtime.restore_usage_reports.assert_called_once_with([manifest["usage_at_checkpoint"]])


def test_dispatch_rejects_non_cumulative_usage_before_provider_load(tmp_path) -> None:
    training_client = _TrainingClient()
    runtime = SimpleNamespace(
        training_client=training_client,
        restore_usage_reports=MagicMock(),
    )
    dispatch = TinkerPolicyDispatch(_cfg(), runtime)

    policy_dir = tmp_path / "global_step_2" / "policy"
    policy_dir.mkdir(parents=True)
    (policy_dir / "tinker_checkpoint.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "provider_path": "tinker://checkpoint-2",
                "global_step": 2,
                "usage_at_checkpoint": {"metrics": {}},
            }
        )
    )

    with pytest.raises(ValueError, match="cumulative current-format usage report"):
        dispatch.load_checkpoint("policy", str(policy_dir))

    assert training_client.loaded_states == []
    runtime.restore_usage_reports.assert_not_called()
