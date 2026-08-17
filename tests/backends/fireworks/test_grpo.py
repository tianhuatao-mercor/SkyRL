import base64
import math

import pytest
import torch

from skyrl.backends.fireworks.grpo import (
    build_tinker_dapo_datums,
    build_tinker_grpo_datums,
    build_tinker_logprob_datums,
    training_batch_to_grpo_datum_specs,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch


def _batch() -> TrainingInputBatch:
    # Row 0 contains token id 0 as real prompt text. Row 1 uses the same value
    # as left padding, proving that attention_mask (not token value) drives
    # unpadding. Response fields have width 3 and are right aligned.
    return TrainingInputBatch(
        {
            "sequences": torch.tensor(
                [
                    [10, 0, 11, 30, 31],
                    [0, 20, 40, 41, 42],
                ],
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1],
                ],
                dtype=torch.long,
            ),
            "response_mask": torch.tensor(
                [
                    [0, 1, 1],
                    [1, 1, 1],
                ],
                dtype=torch.long,
            ),
            "loss_mask": torch.tensor(
                [
                    [0, 1, 0],
                    [0, 1, 1],
                ],
                dtype=torch.float32,
            ),
            "advantages": torch.tensor(
                [
                    [0.0, 2.0, float("nan")],
                    [4.0, 5.0, 6.0],
                ],
                dtype=torch.float32,
            ),
            "rollout_logprobs": torch.tensor(
                [
                    [0.0, -0.1, float("nan")],
                    [-0.2, -0.3, -0.4],
                ],
                dtype=torch.float32,
            ),
        }
    )


def test_training_batch_to_grpo_datum_specs_shifts_and_masks() -> None:
    specs = training_batch_to_grpo_datum_specs(_batch(), max_seq_len=4)

    assert len(specs) == 2
    assert specs[0].model_input_token_ids == (10, 0, 11, 30)
    assert specs[0].target_tokens == (0, 0, 30, 31)
    assert specs[0].rollout_logprobs == pytest.approx((0.0, 0.0, -0.1, 0.0))
    assert specs[0].advantages == pytest.approx((0.0, 0.0, 2.0, 0.0))
    assert specs[0].loss_mask == pytest.approx((0.0, 0.0, 1.0, 0.0))

    assert specs[1].model_input_token_ids == (20, 40, 41)
    assert specs[1].target_tokens == (40, 41, 42)
    assert specs[1].rollout_logprobs == pytest.approx((0.0, -0.3, -0.4))
    assert specs[1].advantages == pytest.approx((0.0, 5.0, 6.0))
    assert specs[1].loss_mask == pytest.approx((0.0, 1.0, 1.0))


@pytest.mark.parametrize("field", ["advantages", "rollout_logprobs"])
def test_training_batch_to_grpo_datum_specs_rejects_non_finite_trainable_values(
    field: str,
) -> None:
    batch = _batch()
    batch[field][0, -2] = math.nan

    with pytest.raises(ValueError, match="non-finite value at trainable response index"):
        training_batch_to_grpo_datum_specs(batch)


def test_training_batch_to_grpo_datum_specs_rejects_non_contiguous_padding() -> None:
    batch = _batch()
    batch["attention_mask"][1] = torch.tensor([0, 1, 0, 1, 1])

    with pytest.raises(ValueError, match="contiguous left padding"):
        training_batch_to_grpo_datum_specs(batch)


def test_training_batch_to_grpo_datum_specs_rejects_length_over_limit() -> None:
    with pytest.raises(ValueError, match="exceeding max_seq_len=3"):
        training_batch_to_grpo_datum_specs(_batch(), max_seq_len=3)


def test_build_tinker_grpo_datums() -> None:
    pytest.importorskip("tinker")

    datums = build_tinker_grpo_datums(_batch(), max_seq_len=4)

    assert [datum.model_input.length for datum in datums] == [4, 3]
    target_tokens = datums[0].loss_fn_inputs["target_tokens"]
    target_data = target_tokens.data if hasattr(target_tokens, "data") else target_tokens
    assert list(target_data) == [0, 0, 30, 31]
    assert datums[0].loss_fn_inputs["target_tokens"].dtype == "int64"
    assert datums[0].loss_fn_inputs["logprobs"].dtype == "float32"
    assert datums[0].loss_fn_inputs["advantages"].dtype == "float32"


def test_build_tinker_logprob_datums_preserves_full_targets_and_response_lengths() -> None:
    pytest.importorskip("tinker")

    datums, response_lengths = build_tinker_logprob_datums(_batch(), max_seq_len=4)

    assert response_lengths == [2, 3]
    first_targets = datums[0].loss_fn_inputs["target_tokens"]
    first_target_data = (
        first_targets.data if hasattr(first_targets, "data") else first_targets
    )
    assert list(first_target_data) == [0, 11, 30, 31]


def test_build_tinker_dapo_datums_uses_old_policy_anchor_and_capped_tis() -> None:
    pytest.importorskip("tinker")
    batch = _batch()
    batch["action_log_probs"] = torch.tensor(
        [
            [0.0, math.log(4.0) - 0.1, 0.0],
            [-0.2, -0.3, math.log(0.5) - 0.4],
        ],
        dtype=torch.float32,
    )

    datums, metrics = build_tinker_dapo_datums(
        batch,
        max_seq_len=4,
        token_tis_ratio_clip_high=2.0,
    )

    first_old = datums[0].loss_fn_inputs["logprobs"]
    first_adv = datums[0].loss_fn_inputs["advantages"]
    old_data = first_old.data if hasattr(first_old, "data") else first_old
    adv_data = first_adv.data if hasattr(first_adv, "data") else first_adv
    assert list(old_data) == pytest.approx([0.0, 0.0, math.log(4.0) - 0.1, 0.0])
    assert list(adv_data) == pytest.approx([0.0, 0.0, 4.0, 0.0])
    assert metrics["tis_token_clip_high_ratio"] == pytest.approx(1 / 3)
    assert metrics["tis_importance_ratio_mean"] == pytest.approx((4.0 + 1.0 + 0.5) / 3)


def test_build_tinker_grpo_datums_attaches_router_replay_rows() -> None:
    pytest.importorskip("tinker")
    batch = _batch()
    batch["rollout_expert_indices"] = torch.zeros(
        (2, 5, 2, 1),
        dtype=torch.uint8,
    )
    batch["rollout_expert_indices"][0, :4, :, 0] = torch.tensor(
        [[1, 2], [3, 4], [5, 6], [7, 8]],
        dtype=torch.uint8,
    )
    batch["rollout_expert_indices"][1, 1:4, :, 0] = torch.tensor(
        [[9, 10], [11, 12], [13, 14]],
        dtype=torch.uint8,
    )
    batch["router_padding_mask"] = torch.tensor(
        [
            [False, False, False, False, True],
            [True, False, False, False, True],
        ]
    )

    datums = build_tinker_grpo_datums(
        batch,
        max_seq_len=4,
        enable_router_replay=True,
    )

    expected = [
        [base64.b64encode(bytes(pair)).decode("ascii") for pair in ([1, 2], [3, 4], [5, 6], [7, 8])],
        [base64.b64encode(bytes(pair)).decode("ascii") for pair in ([9, 10], [11, 12], [13, 14])],
    ]
    assert [datum.model_input.routing_matrices for datum in datums] == expected
