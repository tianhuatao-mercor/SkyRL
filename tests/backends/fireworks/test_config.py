import pytest

from skyrl.train.config import SkyRLTrainConfig, get_config_as_yaml_str
from skyrl.train.utils.utils import validate_cfg


def _valid_fireworks_cfg() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "fireworks"
    cfg.trainer.fireworks.base_model = "accounts/fireworks/models/test"
    cfg.trainer.fireworks.max_seq_len = 4096
    cfg.trainer.fireworks.training_shape_id = "accounts/fireworks/trainingShapes/test"
    cfg.trainer.fireworks.trainer_job_id = "skyrl-smoke-test-trainer"
    cfg.trainer.fireworks.deployment_id = "skyrl-smoke-test-rollout"
    cfg.trainer.policy.model.path = "Test/Tokenizer"
    cfg.trainer.policy.model.lora.rank = 8
    cfg.trainer.policy.optimizer_config.num_warmup_steps = 0
    cfg.trainer.policy.optimizer_config.scheduler = "constant_with_warmup"
    cfg.trainer.algorithm.advantage_estimator = "grpo"
    cfg.trainer.algorithm.policy_loss_type = "rollout_is"
    cfg.trainer.algorithm.use_kl_loss = False
    cfg.trainer.algorithm.use_kl_in_reward = False
    cfg.trainer.critic.model.path = None
    cfg.trainer.placement.colocate_all = False
    cfg.trainer.placement.colocate_policy_ref = False
    cfg.trainer.resume_mode = None
    cfg.trainer.ckpt_interval = -1
    cfg.trainer.hf_save_interval = -1
    cfg.trainer.enable_ray_gpu_monitor = False
    cfg.trainer.logger = "console"
    cfg.generator.inference_engine.backend = "fireworks"
    cfg.generator.inference_engine.run_engines_locally = False
    cfg.generator.inference_engine.enable_ray_prometheus_stats = False
    cfg.generator.sampling_params.max_generate_length = 256
    cfg.generator.eval_sampling_params.max_generate_length = 256
    return cfg


def test_validate_fireworks_grpo_config() -> None:
    validate_cfg(_valid_fireworks_cfg())


def test_validate_fireworks_router_replay_config() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.fireworks.enable_router_replay = True
    cfg.generator.inference_engine.enable_return_routed_experts = True

    validate_cfg(cfg)


@pytest.mark.parametrize(
    ("trainer_enabled", "sampler_enabled"),
    [(True, False), (False, True)],
)
def test_validate_fireworks_requires_router_capture_and_replay_together(
    trainer_enabled: bool,
    sampler_enabled: bool,
) -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.fireworks.enable_router_replay = trainer_enabled
    cfg.generator.inference_engine.enable_return_routed_experts = sampler_enabled

    with pytest.raises(ValueError, match="requires both"):
        validate_cfg(cfg)


def test_validate_fireworks_binary_tv_dppo_prompt_mean_config() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.algorithm.policy_loss_type = "dppo"
    cfg.trainer.algorithm.dppo.dppo_type = "binary_tv"
    cfg.trainer.algorithm.loss_reduction = "prompt_mean"

    validate_cfg(cfg)


def test_validate_fireworks_native_dapo_config_with_warmup() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.algorithm.policy_loss_type = "dapo"
    cfg.trainer.algorithm.loss_reduction = "token_mean"
    cfg.trainer.algorithm.eps_clip_low = 0.2
    cfg.trainer.algorithm.eps_clip_high = 0.28
    cfg.trainer.algorithm.clip_ratio_c = 10.0
    cfg.trainer.algorithm.off_policy_correction.tis_ratio_type = "token"
    cfg.trainer.algorithm.off_policy_correction.token_tis_ratio_clip_high = 2.0
    cfg.trainer.recompute_old_logprobs_per_minibatch = True
    cfg.trainer.policy.optimizer_config.num_warmup_steps = 40

    validate_cfg(cfg)


def test_validate_fireworks_rejects_binary_kl_dppo() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.algorithm.policy_loss_type = "dppo"
    cfg.trainer.algorithm.dppo.dppo_type = "binary_kl"

    with pytest.raises(NotImplementedError, match="binary_tv"):
        validate_cfg(cfg)


def test_validate_fireworks_rejects_non_unit_temperature_dppo() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.algorithm.policy_loss_type = "dppo"
    cfg.trainer.algorithm.temperature = 0.7

    with pytest.raises(NotImplementedError, match="temperature=1.0"):
        validate_cfg(cfg)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("delta_low", -0.1),
        ("delta_low", float("nan")),
        ("delta_high", float("inf")),
        ("delta_high", float("-inf")),
    ],
)
def test_validate_fireworks_rejects_invalid_dppo_thresholds(
    field: str,
    value: float,
) -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.algorithm.policy_loss_type = "dppo"
    setattr(cfg.trainer.algorithm.dppo, field, value)

    with pytest.raises(ValueError, match="finite and non-negative"):
        validate_cfg(cfg)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda correction: setattr(correction, "tis_ratio_type", "token"),
        lambda correction: setattr(correction, "sequence_mask_metric", "geometric"),
        lambda correction: setattr(correction, "outlier_token_is_threshold_low", 0.1),
        lambda correction: setattr(correction, "outlier_token_is_threshold_high", 10.0),
        lambda correction: (
            setattr(correction, "token_mask_is_threshold_low", 0.1),
            setattr(correction, "token_mask_is_threshold_high", 10.0),
        ),
    ],
)
def test_validate_fireworks_rejects_every_active_off_policy_correction_mode(
    mutate,
) -> None:
    cfg = _valid_fireworks_cfg()
    mutate(cfg.trainer.algorithm.off_policy_correction)

    with pytest.raises(NotImplementedError, match="off_policy_correction"):
        validate_cfg(cfg)


def test_validate_fireworks_fully_async_grpo_config() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.fully_async.enabled = True
    cfg.trainer.fully_async.max_staleness_steps = 0
    cfg.trainer.fully_async.num_parallel_generation_workers = 2
    cfg.trainer.train_batch_size = 2
    cfg.trainer.policy_mini_batch_size = 2
    cfg.generator.batched = False

    validate_cfg(cfg)


def test_validate_dedicated_fireworks_grpo_config() -> None:
    cfg = _valid_fireworks_cfg()

    validate_cfg(cfg)


def test_validate_dedicated_full_parameter_fireworks_grpo_config() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.fireworks.training_shape_id = "accounts/fireworks/trainingShapes/qwen3-4b-minimum"
    cfg.trainer.fireworks.replica_count = 4
    cfg.trainer.policy.model.lora.rank = 0

    validate_cfg(cfg)


def test_validate_fireworks_checkpoint_and_resume_config() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.ckpt_interval = 1
    cfg.trainer.resume_mode = "from_path"
    cfg.trainer.resume_path = "/tmp/ckpts/global_step_7"

    validate_cfg(cfg)


def test_validate_fireworks_resume_from_path_requires_path() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.resume_mode = "from_path"

    with pytest.raises(ValueError, match="resume_path is required"):
        validate_cfg(cfg)


def test_validate_dedicated_requires_positive_replica_count() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.fireworks.replica_count = 0

    with pytest.raises(ValueError, match="replica_count > 0"):
        validate_cfg(cfg)


def test_validate_dedicated_requires_positive_trainer_replica_count() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.fireworks.trainer_replica_count = 0

    with pytest.raises(ValueError, match="trainer_replica_count > 0"):
        validate_cfg(cfg)


@pytest.mark.parametrize("seconds", [0, 10801])
def test_validate_fireworks_inactivity_timeout(seconds: int) -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.fireworks.trainer_inactivity_timeout_s = seconds

    with pytest.raises(ValueError, match="between 1 and 10800"):
        validate_cfg(cfg)


def test_validate_dedicated_requires_auditable_resource_ids() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.fireworks.training_shape_id = None

    with pytest.raises(ValueError, match="training_shape_id"):
        validate_cfg(cfg)


def test_validate_fireworks_gpu_hour_cost_config() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.fireworks.billing_gpu_type = "B200"
    cfg.trainer.fireworks.billing_trainer_gpus_per_replica = 4
    cfg.trainer.fireworks.billing_rollout_gpus_per_replica = 4
    cfg.trainer.fireworks.billing_gpu_price_per_hour_usd = 10.0

    validate_cfg(cfg)


def test_validate_fireworks_gpu_hour_cost_config_is_all_or_none() -> None:
    cfg = _valid_fireworks_cfg()
    cfg.trainer.fireworks.billing_gpu_type = "B200"

    with pytest.raises(ValueError, match="cost reporting is all-or-none"):
        validate_cfg(cfg)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda cfg: setattr(cfg.trainer.algorithm, "advantage_estimator", "gae"),
            "GRPO-only",
        ),
        (
            lambda cfg: setattr(cfg.trainer.algorithm, "policy_loss_type", "regular"),
            "policy_loss_type='rollout_is'",
        ),
        (
            lambda cfg: setattr(cfg.trainer.critic.model, "path", "critic"),
            "policy-only",
        ),
        (
            lambda cfg: setattr(cfg.trainer.placement, "colocate_all", True),
            "cannot be colocated",
        ),
    ],
)
def test_validate_fireworks_rejects_out_of_scope_algorithms(mutate, message: str) -> None:
    cfg = _valid_fireworks_cfg()
    mutate(cfg)

    with pytest.raises((ValueError, NotImplementedError), match=message):
        validate_cfg(cfg)


def test_fireworks_api_key_is_not_part_of_serialized_config(monkeypatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "fw_secret_value")

    rendered = get_config_as_yaml_str(_valid_fireworks_cfg())

    assert "fw_secret_value" not in rendered
    assert "FIREWORKS_API_KEY" not in rendered
