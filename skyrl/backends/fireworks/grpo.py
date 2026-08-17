"""Translate SkyRL GRPO batches to Fireworks/Tinker datum layouts.

SkyRL stores full token sequences with left padding while response-level
tensors are right aligned.  Fireworks' built-in ``importance_sampling`` loss
expects next-token-shifted arrays spanning the model input.  Keeping the shape
conversion pure makes it testable without opening a Fireworks session.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import torch

from skyrl.backends.fireworks.router_replay import (
    make_tinker_model_input,
    routing_matrices_for_model_inputs,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch

if TYPE_CHECKING:
    import tinker


@dataclass(frozen=True)
class GRPODatumSpec:
    """Provider-independent contents of one Fireworks GRPO datum.

    Every tuple has ``len(model_input_token_ids)`` entries.  Prompt and masked
    response positions have zero advantage, so they contribute no policy
    gradient to Fireworks' importance-sampling loss.
    """

    model_input_token_ids: tuple[int, ...]
    target_tokens: tuple[int, ...]
    rollout_logprobs: tuple[float, ...]
    advantages: tuple[float, ...]
    loss_mask: tuple[float, ...]
    routing_matrices: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        expected = len(self.model_input_token_ids)
        lengths = {
            "target_tokens": len(self.target_tokens),
            "rollout_logprobs": len(self.rollout_logprobs),
            "advantages": len(self.advantages),
            "loss_mask": len(self.loss_mask),
        }
        mismatched = {name: length for name, length in lengths.items() if length != expected}
        if mismatched:
            raise ValueError(f"GRPO datum fields must all have length {expected}, got {mismatched}")
        if self.routing_matrices is not None and len(self.routing_matrices) != expected:
            raise ValueError(
                "GRPO routing_matrices must have one row per model-input token, "
                f"got {len(self.routing_matrices)} for length {expected}"
            )


def _require_matrix(batch: TrainingInputBatch, name: str) -> torch.Tensor:
    value = batch.get(name)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"Fireworks GRPO requires tensor field {name!r}")
    if value.ndim != 2:
        raise ValueError(f"Fireworks GRPO field {name!r} must be rank 2, got shape {tuple(value.shape)}")
    if value.shape[0] != batch.batch_size:
        raise ValueError(
            f"Fireworks GRPO field {name!r} has batch dimension {value.shape[0]}, expected {batch.batch_size}"
        )
    return value.detach().cpu()


def _right_aligned_length(mask: torch.Tensor, *, field_name: str, row_index: int) -> int:
    present = [bool(value) for value in mask.tolist()]
    count = sum(present)
    expected = [False] * (len(present) - count) + [True] * count
    if present != expected:
        raise ValueError(f"{field_name}[{row_index}] must be right aligned")
    return count


def _unpadded_tokens(
    sequences: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    row_index: int,
) -> list[int]:
    present = [bool(value) for value in attention_mask.tolist()]
    count = sum(present)
    expected = [False] * (len(present) - count) + [True] * count
    if present != expected:
        raise ValueError(f"attention_mask[{row_index}] must describe contiguous left padding")
    return [int(token) for token in sequences[attention_mask.bool()].tolist()]


def training_batch_to_grpo_datum_specs(
    batch: TrainingInputBatch,
    *,
    max_seq_len: int | None = None,
    enable_router_replay: bool = False,
) -> list[GRPODatumSpec]:
    """Convert a SkyRL policy mini-batch into shifted GRPO datum specs.

    The caller must run SkyRL's GRPO advantage estimator and loss-reduction
    scaling first.  This function does no reward or advantage normalization; it
    only removes padding, applies ``loss_mask``, and shifts the token layout.

    Args:
        batch: A policy ``TrainingInputBatch`` containing sequences, masks,
            rollout logprobs, and normalized advantages.
        max_seq_len: Optional Fireworks model-input limit.  The checked length
            is ``len(prompt + response) - 1``, matching the submitted
            ``tinker.ModelInput``.

    Returns:
        One :class:`GRPODatumSpec` per input row, preserving order.
    """

    if batch.batch_size == 0:
        return []

    sequences = _require_matrix(batch, "sequences")
    attention_mask = _require_matrix(batch, "attention_mask")
    response_mask = _require_matrix(batch, "response_mask")
    loss_mask = _require_matrix(batch, "loss_mask")
    advantages = _require_matrix(batch, "advantages")
    rollout_logprobs = _require_matrix(batch, "rollout_logprobs")

    response_width = response_mask.shape[1]
    for name, value in (
        ("loss_mask", loss_mask),
        ("advantages", advantages),
        ("rollout_logprobs", rollout_logprobs),
    ):
        if value.shape[1] != response_width:
            raise ValueError(
                f"Fireworks GRPO response field {name!r} has width {value.shape[1]}, " f"expected {response_width}"
            )

    specs: list[GRPODatumSpec] = []
    for row_index in range(batch.batch_size):
        tokens = _unpadded_tokens(sequences[row_index], attention_mask[row_index], row_index=row_index)
        response_len = _right_aligned_length(response_mask[row_index], field_name="response_mask", row_index=row_index)
        if response_len == 0:
            raise ValueError(f"Fireworks GRPO sample {row_index} has an empty response")

        prompt_len = len(tokens) - response_len
        if prompt_len < 1:
            raise ValueError(
                f"Fireworks GRPO sample {row_index} must contain at least one prompt token; "
                f"got {len(tokens)} total tokens and {response_len} response tokens"
            )

        model_input_token_ids = tokens[:-1]
        if max_seq_len is not None and len(model_input_token_ids) > max_seq_len:
            raise ValueError(
                f"Fireworks GRPO sample {row_index} has model-input length {len(model_input_token_ids)}, "
                f"exceeding max_seq_len={max_seq_len}"
            )

        response_slice = slice(response_width - response_len, response_width)
        response_loss_mask = [float(value) for value in loss_mask[row_index, response_slice].tolist()]
        response_advantages = [float(value) for value in advantages[row_index, response_slice].tolist()]
        response_logprobs = [float(value) for value in rollout_logprobs[row_index, response_slice].tolist()]

        masked_advantages: list[float] = []
        masked_logprobs: list[float] = []
        for token_index, (mask_value, advantage, logprob) in enumerate(
            zip(response_loss_mask, response_advantages, response_logprobs, strict=True)
        ):
            if mask_value not in (0.0, 1.0):
                raise ValueError(
                    f"loss_mask[{row_index}] contains {mask_value} at response index {token_index}; "
                    "Fireworks GRPO requires a binary loss mask"
                )
            if mask_value == 0.0:
                masked_advantages.append(0.0)
                masked_logprobs.append(0.0)
                continue
            if not math.isfinite(advantage):
                raise ValueError(
                    f"advantages[{row_index}] contains a non-finite value at trainable response index {token_index}"
                )
            if not math.isfinite(logprob):
                raise ValueError(
                    f"rollout_logprobs[{row_index}] contains a non-finite value at trainable response index "
                    f"{token_index}"
                )
            masked_advantages.append(advantage)
            masked_logprobs.append(logprob)

        prompt_prediction_count = prompt_len - 1
        specs.append(
            GRPODatumSpec(
                model_input_token_ids=tuple(model_input_token_ids),
                target_tokens=tuple([0] * prompt_prediction_count + tokens[prompt_len:]),
                rollout_logprobs=tuple([0.0] * prompt_prediction_count + masked_logprobs),
                advantages=tuple([0.0] * prompt_prediction_count + masked_advantages),
                loss_mask=tuple([0.0] * prompt_prediction_count + response_loss_mask),
            )
        )

    if enable_router_replay:
        encoded_routes = routing_matrices_for_model_inputs(
            batch,
            [len(spec.model_input_token_ids) for spec in specs],
        )
        specs = [
            replace(spec, routing_matrices=routes)
            for spec, routes in zip(specs, encoded_routes, strict=True)
        ]

    return specs


def _to_tinker_datum(spec: GRPODatumSpec) -> Any:
    try:
        import tinker
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise ImportError(
            "Fireworks GRPO datum construction requires the 'tinker' package; "
            "install SkyRL with the Fireworks/Tinker extra"
        ) from exc

    return tinker.Datum(
        model_input=make_tinker_model_input(
            spec.model_input_token_ids,
            spec.routing_matrices,
        ),
        loss_fn_inputs={
            "target_tokens": tinker.TensorData(data=list(spec.target_tokens), dtype="int64"),
            "logprobs": tinker.TensorData(data=list(spec.rollout_logprobs), dtype="float32"),
            "advantages": tinker.TensorData(data=list(spec.advantages), dtype="float32"),
        },
    )


def build_tinker_grpo_datums(
    batch: TrainingInputBatch,
    *,
    max_seq_len: int | None = None,
    enable_router_replay: bool = False,
) -> list["tinker.Datum"]:
    """Build concrete Tinker datums for Fireworks ``importance_sampling``."""

    return [
        _to_tinker_datum(spec)
        for spec in training_batch_to_grpo_datum_specs(
            batch,
            max_seq_len=max_seq_len,
            enable_router_replay=enable_router_replay,
        )
    ]


def build_tinker_logprob_datums(
    batch: TrainingInputBatch,
    *,
    max_seq_len: int | None = None,
) -> tuple[list["tinker.Datum"], list[int]]:
    """Build cross-entropy datums for a hosted old-policy forward.

    Fireworks returns one target-token logprob for every shifted model-input
    position.  The caller uses the returned response lengths to retain only
    the response suffix and left-pad it back to SkyRL's response tensor width.
    """

    if batch.batch_size == 0:
        return [], []

    try:
        import tinker
    except ImportError as exc:  # pragma: no cover - depends on optional installation
        raise ImportError(
            "Fireworks policy-forward datum construction requires the 'tinker' package; "
            "install SkyRL with the Fireworks/Tinker extra"
        ) from exc

    sequences = _require_matrix(batch, "sequences")
    attention_mask = _require_matrix(batch, "attention_mask")
    response_mask = _require_matrix(batch, "response_mask")
    datums: list[tinker.Datum] = []
    response_lengths: list[int] = []
    for row_index in range(batch.batch_size):
        tokens = _unpadded_tokens(
            sequences[row_index], attention_mask[row_index], row_index=row_index
        )
        response_len = _right_aligned_length(
            response_mask[row_index],
            field_name="response_mask",
            row_index=row_index,
        )
        if response_len == 0:
            raise ValueError(
                f"Fireworks policy-forward sample {row_index} has an empty response"
            )
        model_input_token_ids = tokens[:-1]
        if max_seq_len is not None and len(model_input_token_ids) > max_seq_len:
            raise ValueError(
                f"Fireworks policy-forward sample {row_index} has model-input length "
                f"{len(model_input_token_ids)}, exceeding max_seq_len={max_seq_len}"
            )
        datums.append(
            tinker.Datum(
                model_input=make_tinker_model_input(model_input_token_ids, None),
                loss_fn_inputs={
                    "target_tokens": tinker.TensorData(
                        data=tokens[1:], dtype="int64"
                    ),
                    # Cross-entropy forward returns target-token logprobs.  The
                    # weights do not affect those logits, but the Fireworks
                    # datum schema requires one weight per target position.
                    "weights": tinker.TensorData(
                        data=[0.0] * len(model_input_token_ids), dtype="float32"
                    ),
                },
            )
        )
        response_lengths.append(response_len)
    return datums, response_lengths


def build_tinker_dapo_datums(
    batch: TrainingInputBatch,
    *,
    max_seq_len: int | None = None,
    token_tis_ratio_clip_high: float = 2.0,
    enable_router_replay: bool = False,
) -> tuple[list["tinker.Datum"], dict[str, float]]:
    """Build native DAPO datums with SkyRL's token-level TIS correction.

    The provider ``dapo`` kernel receives old-policy logprobs for its PPO
    ratio.  The separate behavior-policy correction
    ``clip(exp(old_policy - rollout), max=C)`` is detached and folded into
    the already loss-reduction-scaled token advantages, matching SkyRL's
    Megatron DAPO path.
    """

    if not math.isfinite(token_tis_ratio_clip_high) or token_tis_ratio_clip_high <= 0:
        raise ValueError("token_tis_ratio_clip_high must be finite and positive")
    specs = training_batch_to_grpo_datum_specs(
        batch,
        max_seq_len=max_seq_len,
        enable_router_replay=enable_router_replay,
    )
    old_logprobs = _require_matrix(batch, "action_log_probs")
    response_mask = _require_matrix(batch, "response_mask")
    if old_logprobs.shape != response_mask.shape:
        raise ValueError(
            "Fireworks DAPO action_log_probs must align with response_mask; "
            f"got {tuple(old_logprobs.shape)} vs {tuple(response_mask.shape)}"
        )

    datums: list[tinker.Datum] = []
    capped_tokens = 0
    trainable_tokens = 0
    ratio_sum = 0.0
    for row_index, spec in enumerate(specs):
        response_len = _right_aligned_length(
            response_mask[row_index],
            field_name="response_mask",
            row_index=row_index,
        )
        response_width = response_mask.shape[1]
        response_slice = slice(response_width - response_len, response_width)
        old_response = [
            float(value) for value in old_logprobs[row_index, response_slice].tolist()
        ]
        prompt_prediction_count = len(spec.model_input_token_ids) - response_len
        old_full = [0.0] * prompt_prediction_count + old_response
        corrected_advantages: list[float] = []
        for old_lp, rollout_lp, advantage, mask in zip(
            old_full,
            spec.rollout_logprobs,
            spec.advantages,
            spec.loss_mask,
            strict=True,
        ):
            if not mask:
                corrected_advantages.append(0.0)
                continue
            if not math.isfinite(old_lp):
                raise ValueError(
                    f"action_log_probs[{row_index}] contains a non-finite trainable value"
                )
            ratio = math.exp(max(-20.0, min(20.0, old_lp - rollout_lp)))
            capped = min(ratio, token_tis_ratio_clip_high)
            corrected_advantages.append(float(advantage * capped))
            capped_tokens += int(ratio > token_tis_ratio_clip_high)
            trainable_tokens += 1
            ratio_sum += ratio

        dapo_spec = replace(
            spec,
            rollout_logprobs=tuple(old_full),
            advantages=tuple(corrected_advantages),
        )
        datums.append(_to_tinker_datum(dapo_spec))

    denominator = max(trainable_tokens, 1)
    return datums, {
        "tis_token_clip_high_ratio": capped_tokens / denominator,
        "tis_importance_ratio_mean": ratio_sum / denominator,
    }
