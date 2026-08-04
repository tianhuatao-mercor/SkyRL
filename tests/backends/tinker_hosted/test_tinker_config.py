import pytest

from skyrl.train.config import SkyRLTrainConfig, get_config_as_yaml_str
from skyrl.train.utils.utils import validate_cfg


def valid_tinker_cfg() -> SkyRLTrainConfig:
    cfg = SkyRLTrainConfig()
    cfg.trainer.strategy = "tinker"
    cfg.trainer.tinker.base_model = "Qwen/Qwen3.5-4B"
    cfg.trainer.tinker.max_seq_len = 4096
    cfg.trainer.policy.model.path = "Qwen/Qwen3.5-4B"
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
    cfg.generator.inference_engine.backend = "tinker"
    cfg.generator.inference_engine.run_engines_locally = False
    cfg.generator.inference_engine.enable_ray_prometheus_stats = False
    cfg.generator.sampling_params.max_generate_length = 256
    cfg.generator.eval_sampling_params.max_generate_length = 256
    return cfg


def test_validate_sync_tinker_grpo_config() -> None:
    validate_cfg(valid_tinker_cfg())


def test_validate_fully_async_tinker_grpo_config() -> None:
    cfg = valid_tinker_cfg()
    cfg.trainer.fully_async.enabled = True
    cfg.trainer.fully_async.max_staleness_steps = 1
    cfg.trainer.fully_async.num_parallel_generation_workers = 2
    cfg.trainer.train_batch_size = 2
    cfg.trainer.policy_mini_batch_size = 2
    cfg.generator.batched = False

    validate_cfg(cfg)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda cfg: setattr(cfg.trainer.policy.model.lora, "rank", 0),
            "lora.rank > 0",
        ),
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
    ],
)
def test_validate_tinker_rejects_out_of_scope_config(mutate, message) -> None:
    cfg = valid_tinker_cfg()
    mutate(cfg)

    with pytest.raises((ValueError, NotImplementedError), match=message):
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
def test_validate_tinker_rejects_every_active_off_policy_correction_mode(
    mutate,
) -> None:
    cfg = valid_tinker_cfg()
    mutate(cfg.trainer.algorithm.off_policy_correction)

    with pytest.raises(NotImplementedError, match="off_policy_correction"):
        validate_cfg(cfg)


def test_tinker_api_key_is_not_serialized(monkeypatch) -> None:
    monkeypatch.setenv("TINKER_API_KEY", "tinker_secret_value")

    rendered = get_config_as_yaml_str(valid_tinker_cfg())

    assert "tinker_secret_value" not in rendered
    assert "TINKER_API_KEY" not in rendered


def test_validate_tinker_rejects_negative_cost_rate() -> None:
    cfg = valid_tinker_cfg()
    cfg.trainer.tinker.sample_price_per_million_tokens = -1.0

    with pytest.raises(ValueError, match="sample_price_per_million_tokens must be non-negative"):
        validate_cfg(cfg)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "service_bootstrap_max_attempts",
            0,
            "service_bootstrap_max_attempts must be positive",
        ),
        (
            "service_bootstrap_retry_backoff_s",
            -1.0,
            "service_bootstrap_retry_backoff_s must be non-negative",
        ),
    ],
)
def test_validate_tinker_rejects_invalid_bootstrap_retry_config(field, value, message) -> None:
    cfg = valid_tinker_cfg()
    setattr(cfg.trainer.tinker, field, value)

    with pytest.raises(ValueError, match=message):
        validate_cfg(cfg)


def test_validate_tinker_cost_watchdog_requires_positive_limit_and_prices() -> None:
    cfg = valid_tinker_cfg()
    cfg.trainer.tinker.max_estimated_cost_usd = 0

    with pytest.raises(ValueError, match="max_estimated_cost_usd must be positive"):
        validate_cfg(cfg)

    cfg.trainer.tinker.max_estimated_cost_usd = 10
    with pytest.raises(
        ValueError,
        match="max_estimated_cost_usd requires token prices",
    ):
        validate_cfg(cfg)

    cfg.trainer.tinker.prefill_price_per_million_tokens = 1
    cfg.trainer.tinker.sample_price_per_million_tokens = 2
    cfg.trainer.tinker.train_price_per_million_tokens = 3
    validate_cfg(cfg)
