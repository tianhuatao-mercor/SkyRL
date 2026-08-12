"""Client-side binary-TV DPPO for the Fireworks Training API.

Fireworks' custom-loss API returns differentiable current-policy token
log-probabilities to the caller, then sends the resulting gradients back to
the hosted trainer.  That is the narrow primitive needed to preserve SkyRL's
DPPO objective without moving model execution out of Fireworks.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from skyrl.backends.fireworks.grpo import (
    GRPODatumSpec,
    training_batch_to_grpo_datum_specs,
)
from skyrl.backends.fireworks.router_replay import make_tinker_model_input
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch


@dataclass(slots=True)
class _StreamingMoments:
    """Constant-memory summary for token-level diagnostics."""

    count: int = 0
    total: float = 0.0
    square_total: float = 0.0
    minimum: float = math.inf
    maximum: float = -math.inf

    def update(self, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return
        values = values.detach().double()
        self.count += values.numel()
        self.total += values.sum().item()
        self.square_total += values.square().sum().item()
        self.minimum = min(self.minimum, values.min().item())
        self.maximum = max(self.maximum, values.max().item())

    def as_metrics(self, prefix: str) -> dict[str, float]:
        if self.count == 0:
            return {}
        mean = self.total / self.count
        variance = max(0.0, self.square_total / self.count - mean**2)
        return {
            f"{prefix}_mean": mean,
            f"{prefix}_std": math.sqrt(variance),
            f"{prefix}_min": self.minimum,
            f"{prefix}_max": self.maximum,
        }


@dataclass(slots=True)
class _StreamingLogRatioHistogram:
    """Bounded-memory importance-ratio quantiles and cap-hit counts.

    Quantiles are reconstructed from a fixed-width histogram in log-ratio
    space. Cap-hit counts are exact. Keeping the sketch on CPU prevents these
    diagnostics from retaining token-sized accelerator tensors after a row is
    processed.
    """

    bin_count: int = 4096
    lower: float = -20.0
    upper: float = 20.0
    count: int = 0
    cap_5_hits: int = 0
    cap_20_hits: int = 0
    counts: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        if self.bin_count <= 0 or self.lower >= self.upper:
            raise ValueError("Invalid streaming log-ratio histogram bounds")
        self.counts = torch.zeros(self.bin_count, dtype=torch.int64)

    def update(self, log_ratios: torch.Tensor) -> None:
        if log_ratios.numel() == 0:
            return
        values = log_ratios.detach().float().clamp(self.lower, self.upper).cpu()
        self.count += values.numel()
        self.cap_5_hits += int((values > math.log(5.0)).sum().item())
        self.cap_20_hits += int((values > math.log(20.0)).sum().item())
        indices = torch.floor(
            (values - self.lower) * (self.bin_count / (self.upper - self.lower))
        ).long()
        indices.clamp_(0, self.bin_count - 1)
        self.counts += torch.bincount(indices, minlength=self.bin_count)

    def _value_at_rank(self, rank: int) -> float:
        cumulative = self.counts.cumsum(dim=0)
        index = int(
            torch.searchsorted(
                cumulative,
                torch.tensor(rank + 1, dtype=cumulative.dtype),
            ).item()
        )
        log_ratio = self.lower + (index + 0.5) * (
            (self.upper - self.lower) / self.bin_count
        )
        return math.exp(log_ratio)

    def _quantile(self, quantile: float) -> float:
        rank = quantile * (self.count - 1)
        low_rank = math.floor(rank)
        high_rank = math.ceil(rank)
        low = self._value_at_rank(low_rank)
        if low_rank == high_rank:
            return low
        high = self._value_at_rank(high_rank)
        return low + (rank - low_rank) * (high - low)

    def as_metrics(self, prefix: str) -> dict[str, float]:
        if self.count == 0:
            return {}
        metrics = {
            f"{prefix}_p50": self._quantile(0.50),
            f"{prefix}_p90": self._quantile(0.90),
            f"{prefix}_p95": self._quantile(0.95),
            f"{prefix}_p99": self._quantile(0.99),
            f"{prefix}_cap_5_hit_ratio": self.cap_5_hits / self.count,
            f"{prefix}_cap_20_hit_ratio": self.cap_20_hits / self.count,
        }
        return metrics


def make_binary_tv_dppo_loss(
    specs: Sequence[GRPODatumSpec],
    *,
    delta_low: float,
    delta_high: float,
) -> Callable[[list[Any], list[torch.Tensor]], tuple[torch.Tensor, dict[str, float]]]:
    """Return a Fireworks custom loss matching SkyRL binary-TV DPPO.

    ``spec.advantages`` already includes SkyRL's selected loss reduction.  In
    particular, ``prompt_mean`` is applied before the batch reaches this
    function, so summing the token losses preserves that reduction exactly.
    """

    if not specs:
        raise ValueError("Fireworks DPPO requires at least one datum")
    if (
        not math.isfinite(delta_low)
        or not math.isfinite(delta_high)
        or delta_low < 0
        or delta_high < 0
    ):
        raise ValueError("DPPO delta thresholds must be finite and non-negative")

    def loss_fn(
        data: list[Any],
        logprobs_list: list[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if len(data) != len(specs) or len(logprobs_list) != len(specs):
            raise ValueError(
                "Fireworks DPPO datum/logprob counts must match the prepared specs"
            )

        total_loss: torch.Tensor | None = None
        clipped_tokens = 0
        train_logprob_moments = _StreamingMoments()
        rollout_logprob_moments = _StreamingMoments()
        training_advantage_moments = _StreamingMoments()
        logprob_diff_moments = _StreamingMoments()
        abs_logprob_diff_moments = _StreamingMoments()
        importance_ratio_moments = _StreamingMoments()
        retained_ratio_histogram = _StreamingLogRatioHistogram()
        approx_kl_sum = 0.0
        for row_index, (spec, current_logprobs) in enumerate(
            zip(specs, logprobs_list, strict=True)
        ):
            expected = len(spec.model_input_token_ids)
            if current_logprobs.ndim != 1 or current_logprobs.numel() != expected:
                raise ValueError(
                    "Fireworks DPPO current logprobs must be one-dimensional and "
                    f"token-aligned; row {row_index} expected {expected}, got "
                    f"shape {tuple(current_logprobs.shape)}"
                )
            if not torch.isfinite(current_logprobs).all():
                raise ValueError(
                    f"Fireworks DPPO current logprobs contain non-finite values in row {row_index}"
                )

            compute_logprobs = current_logprobs.float()
            behavior_logprobs = torch.tensor(
                spec.rollout_logprobs,
                dtype=compute_logprobs.dtype,
                device=compute_logprobs.device,
            )
            advantages = torch.tensor(
                spec.advantages,
                dtype=compute_logprobs.dtype,
                device=compute_logprobs.device,
            )
            trainable = torch.tensor(
                spec.loss_mask,
                dtype=torch.bool,
                device=compute_logprobs.device,
            )

            logprob_diff = compute_logprobs - behavior_logprobs
            ratio = torch.exp(torch.clamp(logprob_diff, -20.0, 20.0))
            with torch.no_grad():
                probability_delta = torch.exp(compute_logprobs) - torch.exp(
                    behavior_logprobs
                )
                keep = torch.ones_like(advantages)
                keep[(advantages > 0) & (probability_delta > delta_high)] = 0.0
                keep[(advantages < 0) & (-probability_delta > delta_low)] = 0.0

                # The provider computes current-policy logprobs while SkyRL
                # carries the behavior-policy logprobs recorded at rollout.
                # Summarize their token-aligned gap here, where both values
                # are available, without retaining a second token-sized copy
                # after the custom-loss callback returns. Masked prompt and
                # padding positions are deliberately excluded.
                trainable_logprob_diff = logprob_diff[trainable].double()
                if trainable_logprob_diff.numel() > 0:
                    trainable_train_logprobs = compute_logprobs[trainable].double()
                    trainable_rollout_logprobs = behavior_logprobs[trainable].double()
                    trainable_advantages = advantages[trainable].double()
                    trainable_abs_diff = trainable_logprob_diff.abs()
                    trainable_ratio = torch.exp(
                        trainable_logprob_diff.clamp(-20.0, 20.0)
                    )
                    train_logprob_moments.update(trainable_train_logprobs)
                    rollout_logprob_moments.update(trainable_rollout_logprobs)
                    training_advantage_moments.update(trainable_advantages)
                    logprob_diff_moments.update(trainable_logprob_diff)
                    abs_logprob_diff_moments.update(trainable_abs_diff)
                    importance_ratio_moments.update(trainable_ratio)
                    retained_ratio_histogram.update(
                        logprob_diff[trainable & (keep != 0)]
                    )
                    # Schulman-style non-negative approximation to
                    # KL(behavior || current), averaged over action tokens.
                    approx_kl_sum += (
                        (trainable_ratio - 1.0 - trainable_logprob_diff).sum().item()
                    )

            row_loss = -(ratio * advantages * keep).sum()
            total_loss = row_loss if total_loss is None else total_loss + row_loss
            clipped_tokens += int(((keep == 0) & trainable).sum().item())

        assert total_loss is not None
        trainable_tokens = logprob_diff_moments.count
        clip_ratio = clipped_tokens / max(trainable_tokens, 1)
        metrics = {
            "clip_ratio": float(clip_ratio),
            "final_loss": float(total_loss.detach().item()),
        }
        if trainable_tokens > 0:
            metrics.update(train_logprob_moments.as_metrics("train_logprobs"))
            metrics.update(rollout_logprob_moments.as_metrics("rollout_logprobs"))
            metrics.update(training_advantage_moments.as_metrics("training_advantages"))
            metrics.update(
                logprob_diff_moments.as_metrics("rollout_train_logprobs_diff")
            )
            metrics.update(
                abs_logprob_diff_moments.as_metrics("rollout_train_logprobs_abs_diff")
            )
            metrics.update(
                importance_ratio_moments.as_metrics("rollout_train_importance_ratio")
            )
            metrics.update(
                retained_ratio_histogram.as_metrics(
                    "rollout_train_importance_ratio_retained"
                )
            )
            metrics["dppo_retained_token_ratio"] = (
                retained_ratio_histogram.count / trainable_tokens
            )
            metrics["rollout_train_approx_kl"] = approx_kl_sum / trainable_tokens
        return total_loss, metrics

    return loss_fn


def build_tinker_binary_tv_dppo_request(
    batch: TrainingInputBatch,
    *,
    max_seq_len: int | None,
    delta_low: float,
    delta_high: float,
    enable_router_replay: bool = False,
) -> tuple[list[Any], Callable[..., tuple[torch.Tensor, dict[str, float]]]]:
    """Build target-token datums plus the aligned custom DPPO closure."""

    try:
        import tinker
    except ImportError as exc:  # pragma: no cover - optional provider dependency
        raise ImportError(
            "Fireworks DPPO construction requires the 'tinker' package; "
            "install SkyRL with the Fireworks extra"
        ) from exc

    specs = training_batch_to_grpo_datum_specs(
        batch,
        max_seq_len=max_seq_len,
        enable_router_replay=enable_router_replay,
    )
    datums = [
        tinker.Datum(
            model_input=make_tinker_model_input(
                spec.model_input_token_ids,
                spec.routing_matrices,
            ),
            loss_fn_inputs={
                "target_tokens": tinker.TensorData(
                    data=list(spec.target_tokens), dtype="int64"
                )
            },
        )
        for spec in specs
    ]
    return datums, make_binary_tv_dppo_loss(
        specs,
        delta_low=delta_low,
        delta_high=delta_high,
    )
