import math
from types import SimpleNamespace

import pytest
import torch

from skyrl.backends.fireworks.dppo import (
    build_tinker_binary_tv_dppo_request,
    make_binary_tv_dppo_loss,
)
from skyrl.backends.fireworks.grpo import GRPODatumSpec
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.ppo_utils import dppo_policy_loss
from skyrl.train.config import SkyRLTrainConfig


def _spec(
    *,
    behavior_probs: list[float],
    advantages: list[float],
    loss_mask: list[float],
) -> GRPODatumSpec:
    length = len(behavior_probs)
    return GRPODatumSpec(
        model_input_token_ids=tuple(range(10, 10 + length)),
        target_tokens=tuple(range(20, 20 + length)),
        rollout_logprobs=tuple(math.log(value) for value in behavior_probs),
        advantages=tuple(advantages),
        loss_mask=tuple(loss_mask),
    )


def test_binary_tv_custom_loss_matches_expected_loss_mask_and_gradients() -> None:
    specs = [
        _spec(
            behavior_probs=[0.2, 0.2, 0.2],
            advantages=[0.0, 2.0, 3.0],
            loss_mask=[0.0, 1.0, 1.0],
        ),
        _spec(
            behavior_probs=[0.2, 0.2, 0.2],
            advantages=[-4.0, -5.0, 0.0],
            # The final trainable token deliberately has zero advantage. It
            # must remain in the clip-ratio denominator.
            loss_mask=[1.0, 1.0, 1.0],
        ),
    ]
    current = [
        torch.log(torch.tensor([0.9, 0.5, 0.3])).detach().requires_grad_(),
        torch.log(torch.tensor([0.02, 0.1, 0.9])).detach().requires_grad_(),
    ]
    loss_fn = make_binary_tv_dppo_loss(
        specs,
        delta_low=0.15,
        delta_high=0.15,
    )

    loss, metrics = loss_fn([SimpleNamespace(), SimpleNamespace()], current)
    loss.backward()

    reference_current = torch.cat(
        [value.detach() for value in current]
    ).requires_grad_()
    reference_behavior = torch.tensor(
        [value for spec in specs for value in spec.rollout_logprobs]
    )
    reference_advantages = torch.tensor(
        [value for spec in specs for value in spec.advantages]
    )
    reference_mask = torch.tensor([value for spec in specs for value in spec.loss_mask])
    algorithm = SkyRLTrainConfig().trainer.algorithm
    algorithm.dppo.dppo_type = "binary_tv"
    algorithm.dppo.delta_low = 0.15
    algorithm.dppo.delta_high = 0.15
    reference_loss, reference_metrics = dppo_policy_loss(
        reference_current,
        reference_behavior,
        reference_advantages,
        algorithm,
        loss_mask=reference_mask,
        rollout_logprobs=reference_behavior,
    )
    reference_loss.backward()

    assert loss.item() == pytest.approx(-2.0)
    assert metrics["final_loss"] == pytest.approx(-2.0)
    assert metrics["clip_ratio"] == pytest.approx(0.4)
    trainable_current = torch.cat([current[0].detach()[1:], current[1].detach()])
    trainable_behavior = torch.full_like(trainable_current, math.log(0.2))
    trainable_advantages = torch.tensor([2.0, 3.0, -4.0, -5.0, 0.0])
    # The custom loss intentionally computes the difference in float32, then
    # uses float64 only for numerically stable summary accumulation.
    logprob_diff = (trainable_current - trainable_behavior).double()
    abs_logprob_diff = logprob_diff.abs()
    importance_ratio = torch.exp(logprob_diff.clamp(-20.0, 20.0))
    for prefix, values in (
        ("train_logprobs", trainable_current.double()),
        ("rollout_logprobs", trainable_behavior.double()),
        ("training_advantages", trainable_advantages.double()),
    ):
        assert metrics[f"{prefix}_mean"] == pytest.approx(values.mean().item())
        assert metrics[f"{prefix}_std"] == pytest.approx(values.std(correction=0).item())
        assert metrics[f"{prefix}_min"] == pytest.approx(values.min().item())
        assert metrics[f"{prefix}_max"] == pytest.approx(values.max().item())
    assert metrics["rollout_train_logprobs_diff_mean"] == pytest.approx(
        logprob_diff.mean().item()
    )
    assert metrics["rollout_train_logprobs_diff_std"] == pytest.approx(
        logprob_diff.std(correction=0).item()
    )
    assert metrics["rollout_train_logprobs_diff_min"] == pytest.approx(
        logprob_diff.min().item()
    )
    assert metrics["rollout_train_logprobs_diff_max"] == pytest.approx(
        logprob_diff.max().item()
    )
    assert metrics["rollout_train_logprobs_abs_diff_mean"] == pytest.approx(
        abs_logprob_diff.mean().item()
    )
    assert metrics["rollout_train_logprobs_abs_diff_std"] == pytest.approx(
        abs_logprob_diff.std(correction=0).item()
    )
    assert metrics["rollout_train_logprobs_abs_diff_min"] == pytest.approx(
        abs_logprob_diff.min().item()
    )
    assert metrics["rollout_train_logprobs_abs_diff_max"] == pytest.approx(
        abs_logprob_diff.max().item()
    )
    assert metrics["rollout_train_importance_ratio_mean"] == pytest.approx(
        importance_ratio.mean().item()
    )
    assert metrics["rollout_train_importance_ratio_std"] == pytest.approx(
        importance_ratio.std(correction=0).item()
    )
    assert metrics["rollout_train_importance_ratio_min"] == pytest.approx(
        importance_ratio.min().item()
    )
    assert metrics["rollout_train_importance_ratio_max"] == pytest.approx(
        importance_ratio.max().item()
    )
    assert metrics["rollout_train_approx_kl"] == pytest.approx(
        (importance_ratio - 1.0 - logprob_diff).mean().item()
    )
    assert current[0].grad.tolist() == pytest.approx([0.0, 0.0, -4.5])
    assert current[1].grad.tolist() == pytest.approx([0.0, 2.5, 0.0])
    assert loss.item() == pytest.approx(reference_loss.item())
    assert metrics["clip_ratio"] == pytest.approx(reference_metrics["clip_ratio"])
    assert torch.cat([value.grad for value in current]).tolist() == pytest.approx(
        reference_current.grad.tolist()
    )


def test_binary_tv_custom_loss_uses_strict_delta_thresholds() -> None:
    spec = _spec(
        behavior_probs=[0.2, 0.2, 0.2, 0.2],
        advantages=[1.0, 1.0, -1.0, -1.0],
        loss_mask=[1.0, 1.0, 1.0, 1.0],
    )
    current = (
        torch.log(torch.tensor([0.349, 0.351, 0.051, 0.049])).detach().requires_grad_()
    )
    loss_fn = make_binary_tv_dppo_loss(
        [spec],
        delta_low=0.15,
        delta_high=0.15,
    )

    loss, metrics = loss_fn([SimpleNamespace()], [current])
    loss.backward()

    assert loss.item() == pytest.approx(-1.49, abs=1e-5)
    assert metrics["clip_ratio"] == pytest.approx(0.5)
    assert current.grad.tolist() == pytest.approx([-1.745, 0.0, 0.255, 0.0], abs=1e-5)


@pytest.mark.parametrize(
    ("delta_low", "delta_high"),
    [
        (-0.1, 0.15),
        (0.15, -0.1),
        (float("nan"), 0.15),
        (0.15, float("inf")),
        (0.15, float("-inf")),
    ],
)
def test_binary_tv_custom_loss_rejects_invalid_delta_thresholds(
    delta_low: float,
    delta_high: float,
) -> None:
    spec = _spec(
        behavior_probs=[0.2],
        advantages=[1.0],
        loss_mask=[1.0],
    )

    with pytest.raises(ValueError, match="finite and non-negative"):
        make_binary_tv_dppo_loss(
            [spec],
            delta_low=delta_low,
            delta_high=delta_high,
        )


@pytest.mark.parametrize(
    ("data_count", "current", "message"),
    [
        (0, [torch.zeros(2)], "counts must match"),
        (1, [torch.zeros(3)], "row 0 expected 2"),
        (1, [torch.tensor([0.0, float("nan")])], "non-finite values in row 0"),
    ],
)
def test_binary_tv_custom_loss_rejects_misaligned_or_non_finite_results(
    data_count: int,
    current: list[torch.Tensor],
    message: str,
) -> None:
    spec = _spec(
        behavior_probs=[0.2, 0.2],
        advantages=[1.0, -1.0],
        loss_mask=[1.0, 1.0],
    )
    loss_fn = make_binary_tv_dppo_loss(
        [spec],
        delta_low=0.15,
        delta_high=0.15,
    )

    with pytest.raises(ValueError, match=message):
        loss_fn([SimpleNamespace()] * data_count, current)


def test_build_dppo_request_strips_callback_only_inputs_from_provider_datums() -> None:
    pytest.importorskip("tinker")
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor([[10, 11, 20, 21]], dtype=torch.long),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
            "loss_mask": torch.tensor([[1.0, 0.0]]),
            "advantages": torch.tensor([[2.0, float("nan")]]),
            "rollout_logprobs": torch.tensor([[-0.1, float("nan")]]),
        }
    )

    datums, _ = build_tinker_binary_tv_dppo_request(
        batch,
        max_seq_len=3,
        delta_low=0.15,
        delta_high=0.15,
    )

    assert len(datums) == 1
    assert set(datums[0].loss_fn_inputs) == {"target_tokens"}
    # Prompt predictions use a masked placeholder target; response targets
    # remain aligned to the behavior logprobs and advantages in the closure.
    assert list(datums[0].loss_fn_inputs["target_tokens"].data) == [0, 20, 21]
