import base64

import numpy as np
import pytest
import torch

from skyrl.backends.fireworks.router_replay import (
    decode_fireworks_routing_matrices,
    routing_matrices_for_model_inputs,
)
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.train.dataset.preprocess import (
    convert_prompts_responses_to_batch_tensors,
    make_router_padding_mask,
)


def _encode(payload: bytes) -> str:
    return base64.b64encode(payload).decode("ascii")


def _staged_batch(encoded_rows: list[str]) -> TrainingInputBatch:
    routes = decode_fireworks_routing_matrices(encoded_rows)
    (
        sequences,
        attention_mask,
        response_mask,
        rewards,
        loss_mask,
        rollout_logprobs,
        route_tensor,
    ) = convert_prompts_responses_to_batch_tensors(
        0,
        prompts=[[10, 11]],
        responses=[[20, 21]],
        rewards=[[0.0, 1.0]],
        loss_masks=[[1, 1]],
        logprobs=[[-0.2, -0.3]],
        rollout_expert_indices=[routes],
    )
    assert route_tensor is not None
    return TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "rewards": rewards,
            "loss_mask": loss_mask,
            "rollout_logprobs": rollout_logprobs,
            "rollout_expert_indices": route_tensor,
            "router_padding_mask": make_router_padding_mask(
                attention_mask,
                [len(encoded_rows)],
            ),
        }
    )


def test_fireworks_routing_rows_survive_skyrl_padding_byte_for_byte() -> None:
    encoded = [
        _encode(bytes([1, 2, 3, 4, 5, 6])),
        _encode(bytes([7, 8, 9, 10, 11, 12])),
        _encode(bytes([13, 14, 15, 16, 17, 18])),
    ]

    recovered = routing_matrices_for_model_inputs(
        _staged_batch(encoded),
        [3],
    )

    assert recovered == [tuple(encoded)]


def test_decode_fireworks_routing_rows_rejects_bad_or_ragged_payloads() -> None:
    with pytest.raises(ValueError, match="valid base64"):
        decode_fireworks_routing_matrices(["not base64!"])
    with pytest.raises(ValueError, match="consistent decoded size"):
        decode_fireworks_routing_matrices([_encode(b"abc"), _encode(b"abcd")])


def test_router_replay_rejects_uncaptured_model_input_suffix() -> None:
    encoded = [_encode(bytes([1, 2])), _encode(bytes([3, 4]))]

    with pytest.raises(ValueError, match="expected exactly the first 3"):
        routing_matrices_for_model_inputs(_staged_batch(encoded), [3])


def test_decode_uses_writable_compact_uint8_array() -> None:
    routes = decode_fireworks_routing_matrices([_encode(bytes([255, 0, 17]))])

    assert routes.dtype == np.uint8
    assert routes.shape == (1, 3, 1)
    assert routes.flags.writeable


def test_completion_only_routes_are_padded_over_prompt_and_masked_context() -> None:
    route_a = _encode(bytes([1, 2, 3]))
    route_b = _encode(bytes([4, 5, 6]))
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor([[10, 11, 20, 30, 21]]),
            "attention_mask": torch.ones((1, 5), dtype=torch.bool),
            "response_mask": torch.tensor([[1, 1, 1]], dtype=torch.bool),
            "loss_mask": torch.tensor([[1, 0, 1]], dtype=torch.float32),
        }
    )
    batch.metadata = {
        "rollout_routing_matrices": [[route_a, "", route_b]],
    }

    recovered = routing_matrices_for_model_inputs(batch, [4])

    assert recovered == [("", route_a, "", route_b)]


def test_completion_only_routes_require_every_trainable_response_route() -> None:
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor([[10, 20, 21]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.bool),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.bool),
            "loss_mask": torch.tensor([[1, 1]], dtype=torch.float32),
        }
    )
    batch.metadata = {"rollout_routing_matrices": [[_encode(b"abc"), ""]]}

    with pytest.raises(ValueError, match="missing a route for trainable"):
        routing_matrices_for_model_inputs(batch, [2])


def test_completion_only_routes_reject_response_length_mismatch() -> None:
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor([[10, 20, 21]]),
            "attention_mask": torch.ones((1, 3), dtype=torch.bool),
            "response_mask": torch.tensor([[1, 1]], dtype=torch.bool),
            "loss_mask": torch.tensor([[1, 1]], dtype=torch.float32),
        }
    )
    batch.metadata = {"rollout_routing_matrices": [[_encode(b"abc")]]}

    with pytest.raises(ValueError, match="response-token aligned"):
        routing_matrices_for_model_inputs(batch, [2])
