import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import ray
import torch

from skyrl.backends.fireworks.training_backend import FireworksPolicyDispatch
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.config import SkyRLTrainConfig
from skyrl.train.utils.trainer_utils import run_on_each_node


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
        self.forward_backward_custom_calls = []
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

    def forward_backward_custom(
        self,
        datums,
        loss_fn,
        *,
        loss_type_input,
    ):
        self.forward_backward_custom_calls.append((datums, loss_fn, loss_type_input))
        return _Future(
            SimpleNamespace(
                metrics={"final_loss": -2.0, "clip_ratio": 0.4},
                loss_fn_output_type="scalar",
            )
        )

    def optim_step(self, params):
        self.optim_params.append(params)
        return _Future(SimpleNamespace(metrics={"grad_norm": 0.75}))

    def save_state(self, name):
        self.saved_states.append(name)
        return _Future(SimpleNamespace(path=f"tinker://source/weights/{name}"))

    def resolve_checkpoint_path(self, checkpoint_name, source_job_id=None):
        return f"cross-job://{source_job_id}/{checkpoint_name}"

    def load_state_with_optimizer(self, path):
        self.loaded_states.append((path, True))
        return _Future(SimpleNamespace())

    def load_state(self, path):
        self.loaded_states.append((path, False))
        return _Future(SimpleNamespace())


class _GradNormMetricsTrainingClient(_TrainingClient):
    def __init__(self):
        super().__init__()
        self.emit_grad_norm_metrics = None

    def optim_step(self, params, *, emit_grad_norm_metrics=None):
        self.optim_params.append(params)
        self.emit_grad_norm_metrics = emit_grad_norm_metrics
        return _Future(
            SimpleNamespace(
                metrics={
                    "skyrl.ai/grad_norm": 1.5,
                    "skyrl.ai/grad_norm_rms": 0.25,
                }
            )
        )


class _PatchedAdamParamsTrainingClient(_TrainingClient):
    def optim_step(self, params):
        self.optim_params.append(params)
        return _Future(SimpleNamespace(metrics={}))


def _cfg() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.fireworks.base_model = "accounts/fireworks/models/test-base"
    cfg.trainer.fireworks.training_shape_id = "accounts/fireworks/trainingShapes/test-shape"
    cfg.trainer.fireworks.max_seq_len = 128
    cfg.trainer.policy.model.path = "Test/Tokenizer"
    cfg.trainer.policy.model.lora.rank = 8
    cfg.trainer.policy.model.lora.alpha = 16
    cfg.trainer.algorithm.policy_loss_type = "rollout_is"
    return cfg


def test_policy_dispatch_stages_and_submits_importance_sampling(monkeypatch) -> None:
    training_client = _TrainingClient()
    runtime = SimpleNamespace(
        training_client=training_client,
        publish_sampler_weights=None,
        record_forward_backward=MagicMock(),
    )
    built_batches = []
    monotonic_values = iter((100.0, 102.5))
    monkeypatch.setattr(
        "skyrl.backends.fireworks.training_backend.time.monotonic",
        lambda: next(monotonic_values),
    )

    def datum_builder(batch, *, max_seq_len):
        built_batches.append((batch, max_seq_len))
        return ["datum"] * batch.batch_size

    dispatch = FireworksPolicyDispatch(_cfg(), runtime, datum_builder=datum_builder)
    batch = TrainingInputBatch(
        {
            "sequences": torch.arange(12).reshape(4, 3),
            "attention_mask": torch.ones((4, 3), dtype=torch.bool),
        }
    )
    staged = dispatch.stage_data("policy", batch, [(0, 2), (2, 4)])

    assert [part.batch_size for part in staged] == [2, 2]
    output = dispatch.forward_backward_from_staged("policy", staged[0])

    assert training_client.forward_backward_calls == [(["datum", "datum"], "importance_sampling")]
    assert built_batches[0][1] == 128
    assert output.metrics["final_loss"] == pytest.approx(1.25)
    assert output.metrics["response_tokens"] == pytest.approx(4.0)
    runtime.record_forward_backward.assert_called_once_with(
        training_tokens=6,
        elapsed_s=pytest.approx(2.5),
        succeeded=True,
    )


def test_policy_dispatch_submits_binary_tv_dppo_custom_loss(monkeypatch) -> None:
    cfg = _cfg()
    cfg.trainer.algorithm.policy_loss_type = "dppo"
    cfg.trainer.algorithm.dppo.delta_low = 0.15
    cfg.trainer.algorithm.dppo.delta_high = 0.15
    training_client = _TrainingClient()
    runtime = SimpleNamespace(
        training_client=training_client,
        record_forward_backward=MagicMock(),
    )
    custom_loss = object()
    built_requests = []

    def dppo_request_builder(
        batch,
        *,
        max_seq_len,
        delta_low,
        delta_high,
    ):
        built_requests.append((batch, max_seq_len, delta_low, delta_high))
        return ["dppo-datum"] * batch.batch_size, custom_loss

    monotonic_values = iter((50.0, 53.0))
    monkeypatch.setattr(
        "skyrl.backends.fireworks.training_backend.time.monotonic",
        lambda: next(monotonic_values),
    )
    dispatch = FireworksPolicyDispatch(
        cfg,
        runtime,
        dppo_request_builder=dppo_request_builder,
    )
    batch = TrainingInputBatch(
        {
            "sequences": torch.arange(6).reshape(2, 3),
            "attention_mask": torch.ones((2, 3), dtype=torch.bool),
        }
    )

    output = dispatch.forward_backward_from_staged("policy", batch)

    assert training_client.forward_backward_calls == []
    assert training_client.forward_backward_custom_calls == [(["dppo-datum", "dppo-datum"], custom_loss, "logprobs")]
    assert built_requests == [(batch, 128, 0.15, 0.15)]
    assert output.metrics == {"final_loss": -2.0, "clip_ratio": 0.4}
    runtime.record_forward_backward.assert_called_once_with(
        training_tokens=6,
        elapsed_s=pytest.approx(3.0),
        succeeded=True,
    )


def test_policy_dispatch_optimizer_uses_skyrl_optimizer_config() -> None:
    cfg = _cfg()
    cfg.trainer.policy.optimizer_config.lr = 2e-5
    cfg.trainer.policy.optimizer_config.adam_betas = [0.8, 0.9]
    cfg.trainer.policy.optimizer_config.weight_decay = 0.1
    cfg.trainer.policy.optimizer_config.max_grad_norm = 2.0
    training_client = _TrainingClient()
    dispatch = FireworksPolicyDispatch(cfg, SimpleNamespace(training_client=training_client))

    grad_norm = dispatch.optim_step("policy")

    params = training_client.optim_params[0]
    assert params.learning_rate == pytest.approx(2e-5)
    assert params.beta1 == pytest.approx(0.8)
    assert params.beta2 == pytest.approx(0.9)
    assert params.weight_decay == pytest.approx(0.1)
    assert params.grad_clip_norm == pytest.approx(2.0)
    assert grad_norm == pytest.approx(0.75)


def test_policy_dispatch_opts_in_and_exposes_provider_optimizer_metrics() -> None:
    cfg = _cfg()
    cfg.trainer.fireworks.emit_grad_norm_metrics = "detailed"
    training_client = _GradNormMetricsTrainingClient()
    dispatch = FireworksPolicyDispatch(
        cfg,
        SimpleNamespace(training_client=training_client),
    )

    grad_norm = dispatch.optim_step("policy")

    assert training_client.emit_grad_norm_metrics == "detailed"
    assert grad_norm == pytest.approx(1.5)
    assert dispatch.take_last_optimizer_metrics() == {
        "skyrl.ai/grad_norm": 1.5,
        "skyrl.ai/grad_norm_rms": 0.25,
    }
    assert dispatch.take_last_optimizer_metrics() == {}


def test_policy_dispatch_does_not_report_rms_as_global_grad_norm() -> None:
    cfg = _cfg()
    training_client = _GradNormMetricsTrainingClient()
    dispatch = FireworksPolicyDispatch(
        cfg,
        SimpleNamespace(training_client=training_client),
    )
    training_client.optim_step = lambda params, **kwargs: _Future(
        SimpleNamespace(metrics={"skyrl.ai/grad_norm_rms": 0.25})
    )

    assert dispatch.optim_step("policy") is None
    assert dispatch.take_last_optimizer_metrics() == {
        "skyrl.ai/grad_norm_rms": 0.25,
    }


def test_policy_dispatch_uses_patched_adam_field_for_old_optimizer_signature() -> None:
    cfg = _cfg()
    cfg.trainer.fireworks.emit_grad_norm_metrics = "basic"
    training_client = _PatchedAdamParamsTrainingClient()
    dispatch = FireworksPolicyDispatch(
        cfg,
        SimpleNamespace(training_client=training_client),
    )

    dispatch.optim_step("policy")

    params = training_client.optim_params[0]
    if "emit_grad_norm_metrics" in type(params).model_fields:
        assert params.emit_grad_norm_metrics == "basic"


def test_policy_dispatch_finalizes_synchronous_checkpoint_saves() -> None:
    dispatch = FireworksPolicyDispatch(_cfg(), SimpleNamespace(training_client=_TrainingClient()))

    assert dispatch.finalize_pending_saves("policy") is None
    with pytest.raises(NotImplementedError, match="policy-only"):
        dispatch.finalize_pending_saves("critic")


def test_policy_dispatch_saves_and_cross_job_loads_dcp_checkpoint(tmp_path) -> None:
    training_client = _TrainingClient()
    usage_report = {
        "cumulative_across_resumes": True,
        "started_at_utc": "2026-07-30T01:00:00+00:00",
        "billing_stages": [],
        "metrics": {
            "fireworks/usage/training_tokens_total": 120,
            "fireworks/usage/forward_backward_seconds_total": 2.5,
            "fireworks/estimated_cost/gpu_total_usd": 12.5,
        },
    }
    runtime = SimpleNamespace(
        training_client=training_client,
        trainer_job_id="source-trainer",
        usage_report=lambda: usage_report,
        restore_usage_reports=MagicMock(),
    )
    cfg = _cfg()
    cfg.trainer.fireworks.request_timeout_s = 123
    dispatch = FireworksPolicyDispatch(cfg, runtime)
    ckpt_dir = tmp_path / "global_step_7" / "policy"

    dispatch.save_checkpoint("policy", str(ckpt_dir), tokenizer="unused")

    assert len(training_client.saved_states) == 1
    assert training_client.saved_states[0].startswith("skyrl-step-7-")
    manifest = json.loads((ckpt_dir / "fireworks_checkpoint.json").read_text())
    assert manifest["format_version"] == 2
    assert manifest["training_identity"] == {
        "base_model": "accounts/fireworks/models/test-base",
        "training_shape_id": "accounts/fireworks/trainingShapes/test-shape",
        "tokenizer_model": "Test/Tokenizer",
        "lora_rank": 8,
        "lora_alpha": 16,
    }
    assert manifest["source_trainer_job_id"] == "source-trainer"
    assert manifest["cross_job_checkpoint_name"] == "step-7"
    assert manifest["includes_optimizer_state"] is True
    assert manifest["global_step"] == 7
    assert manifest["usage_at_checkpoint"]["metrics"]["fireworks/estimated_cost/gpu_total_usd"] == pytest.approx(12.5)

    dispatch.load_checkpoint(
        "policy",
        str(ckpt_dir),
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
    )

    assert training_client.loaded_states == [
        (
            "cross-job://source-trainer/step-7",
            True,
        )
    ]
    runtime.restore_usage_reports.assert_called_once_with([usage_report])


@pytest.mark.parametrize(
    ("field", "checkpoint_value"),
    [
        ("base_model", "accounts/fireworks/models/different"),
        (
            "training_shape_id",
            "accounts/fireworks/trainingShapes/different",
        ),
        ("tokenizer_model", "Different/Tokenizer"),
        ("lora_rank", 0),
        ("lora_alpha", 32),
    ],
)
def test_policy_dispatch_rejects_checkpoint_method_mismatch_before_provider_load(
    tmp_path,
    field: str,
    checkpoint_value,
) -> None:
    training_client = _TrainingClient()
    dispatch = FireworksPolicyDispatch(
        _cfg(),
        SimpleNamespace(training_client=training_client, trainer_job_id="new-trainer"),
    )
    identity = dispatch._checkpoint_identity()
    identity[field] = checkpoint_value
    ckpt_dir = tmp_path / "global_step_2" / "policy"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "fireworks_checkpoint.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "checkpoint_kind": "fireworks_dcp",
                "training_identity": identity,
                "checkpoint_name": "step-2",
                "provider_path": "tinker://prior-run/weights/step-2",
            }
        )
    )

    with pytest.raises(ValueError, match=field):
        dispatch.load_checkpoint("policy", str(ckpt_dir))

    assert training_client.loaded_states == []


def test_policy_dispatch_accepts_new_version_of_same_training_shape() -> None:
    cfg = _cfg()
    cfg.trainer.fireworks.training_shape_id = (
        "accounts/fireworks/trainingShapes/test-shape/versions/new"
    )
    dispatch = FireworksPolicyDispatch(
        cfg,
        SimpleNamespace(),
    )
    identity = dispatch._checkpoint_identity()
    identity["training_shape_id"] = (
        "accounts/fireworks/trainingShapes/test-shape/versions/old"
    )
    dispatch._preflight_checkpoint_identity(
        {
            "format_version": 2,
            "training_identity": identity,
        },
        manifest_path="fireworks_checkpoint.json",
    )


def test_policy_dispatch_rejects_v2_checkpoint_without_identity_before_load(
    tmp_path,
) -> None:
    training_client = _TrainingClient()
    dispatch = FireworksPolicyDispatch(_cfg(), SimpleNamespace(training_client=training_client, trainer_job_id=None))
    ckpt_dir = tmp_path / "global_step_2" / "policy"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "fireworks_checkpoint.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "checkpoint_kind": "fireworks_dcp",
                "provider_path": "tinker://prior-run/weights/step-2",
            }
        )
    )

    with pytest.raises(ValueError, match="requires training_identity"):
        dispatch.load_checkpoint("policy", str(ckpt_dir))

    assert training_client.loaded_states == []


def test_policy_dispatch_ignores_lora_alpha_for_full_finetune_checkpoint(
    tmp_path,
) -> None:
    training_client = _TrainingClient()
    cfg = _cfg()
    cfg.trainer.policy.model.lora.rank = 0
    cfg.trainer.policy.model.lora.alpha = 16
    usage_report = {
        "cumulative_across_resumes": True,
        "billing_stages": [],
        "metrics": {},
    }
    runtime = SimpleNamespace(
        training_client=training_client,
        trainer_job_id="source-trainer",
        usage_report=lambda: usage_report,
        restore_usage_reports=MagicMock(),
    )
    dispatch = FireworksPolicyDispatch(
        cfg,
        runtime,
    )
    ckpt_dir = tmp_path / "global_step_2" / "policy"
    dispatch.save_checkpoint("policy", str(ckpt_dir))
    manifest_path = ckpt_dir / "fireworks_checkpoint.json"
    manifest = json.loads(manifest_path.read_text())
    assert manifest["training_identity"]["lora_alpha"] is None

    # Alpha is semantically inert when both the checkpoint and current run are
    # full-parameter training.
    manifest["training_identity"]["lora_alpha"] = 999
    manifest_path.write_text(json.dumps(manifest))

    dispatch.load_checkpoint("policy", str(ckpt_dir))

    assert training_client.loaded_states == [("cross-job://source-trainer/step-2", True)]
    runtime.restore_usage_reports.assert_called_once_with([usage_report])


def test_policy_dispatch_rejects_old_manifest_before_provider_load(
    tmp_path,
) -> None:
    training_client = _TrainingClient()
    dispatch = FireworksPolicyDispatch(_cfg(), SimpleNamespace(training_client=training_client, trainer_job_id=None))
    ckpt_dir = tmp_path / "global_step_2" / "policy"
    ckpt_dir.mkdir(parents=True)
    provider_path = "tinker://prior-run/weights/step-2"
    (ckpt_dir / "fireworks_checkpoint.json").write_text(
        json.dumps(
            {
                "format_version": 1,
                "checkpoint_kind": "fireworks_dcp",
                "provider_path": provider_path,
                "usage_at_checkpoint": {"cumulative_across_resumes": True},
            }
        )
    )

    with pytest.raises(ValueError, match="Unsupported Fireworks checkpoint manifest"):
        dispatch.load_checkpoint("policy", str(ckpt_dir))

    assert training_client.loaded_states == []


def test_policy_dispatch_rejects_current_manifest_without_canonical_checkpoint_reference(
    tmp_path,
) -> None:
    training_client = _TrainingClient()
    runtime = SimpleNamespace(
        training_client=training_client,
        trainer_job_id="new-trainer",
    )
    dispatch = FireworksPolicyDispatch(_cfg(), runtime)
    ckpt_dir = tmp_path / "global_step_2" / "policy"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "fireworks_checkpoint.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "checkpoint_kind": "fireworks_dcp",
                "training_identity": dispatch._checkpoint_identity(),
                "checkpoint_name": "skyrl-step-2-label",
                "provider_path": "skyrl-step-2-label",
                "source_trainer_job_id": "source-trainer",
                "includes_optimizer_state": True,
                "global_step": 2,
                "usage_at_checkpoint": {
                    "cumulative_across_resumes": True,
                    "billing_stages": [],
                    "metrics": {},
                },
            }
        )
    )

    with pytest.raises(ValueError, match="cross_job_checkpoint_name"):
        dispatch.load_checkpoint("policy", str(ckpt_dir))

    assert training_client.loaded_states == []


def test_policy_dispatch_rejects_checkpoint_without_current_usage_before_load(tmp_path) -> None:
    training_client = _TrainingClient()
    runtime = SimpleNamespace(training_client=training_client, trainer_job_id=None)
    dispatch = FireworksPolicyDispatch(_cfg(), runtime)
    ckpt_dir = tmp_path / "global_step_2" / "policy"
    ckpt_dir.mkdir(parents=True)
    (ckpt_dir / "fireworks_checkpoint.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "checkpoint_kind": "fireworks_dcp",
                "training_identity": dispatch._checkpoint_identity(),
                "checkpoint_name": "step-2",
                "provider_path": "tinker://prior-run/weights/step-2",
                "source_trainer_job_id": "source-trainer",
                "cross_job_checkpoint_name": "step-2",
                "includes_optimizer_state": True,
            }
        )
    )

    with pytest.raises(ValueError, match="cumulative current-format usage report"):
        dispatch.load_checkpoint("policy", str(ckpt_dir))

    assert training_client.loaded_states == []


def test_hosted_empty_node_cleanup_does_not_initialize_ray(monkeypatch) -> None:
    monkeypatch.setattr(
        ray,
        "remote",
        lambda *args, **kwargs: pytest.fail("ray.remote should not be called"),
    )

    assert run_on_each_node([], lambda: None) == []
