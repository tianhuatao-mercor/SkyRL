"""Fireworks MoE router-replay payload conversion.

Fireworks exposes one base64-encoded ``uint8`` routing matrix per model-input
position.  SkyRL's generic rollout path carries routed experts as an integer
``[tokens, layers, topk]`` array.  The provider's inner matrix dimensions are
not needed to replay it, so this adapter treats each decoded row as opaque
bytes and uses ``[tokens, payload_bytes, 1]`` while the value crosses SkyRL.
Only this module decodes or re-encodes that provider representation.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch


def decode_fireworks_routing_matrices(
    encoded_rows: Sequence[str],
) -> np.ndarray:
    """Decode Fireworks rows into SkyRL's routed-expert array convention.

    The returned shape is ``[tokens, payload_bytes, 1]``.  Flattening either
    inner layout yields the exact provider bytes, which is the only property
    needed by Fireworks training's router-replay API.
    """

    if not encoded_rows:
        raise ValueError("Fireworks router replay returned no routing matrices")

    decoded: list[np.ndarray] = []
    payload_width: int | None = None
    for row_index, encoded in enumerate(encoded_rows):
        if not isinstance(encoded, str) or not encoded:
            raise ValueError(f"Fireworks routing matrix {row_index} must be a non-empty base64 string")
        try:
            payload = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"Fireworks routing matrix {row_index} is not valid base64") from exc
        if not payload:
            raise ValueError(f"Fireworks routing matrix {row_index} decoded to an empty payload")
        if payload_width is None:
            payload_width = len(payload)
        elif len(payload) != payload_width:
            raise ValueError(
                "Fireworks routing matrices must have a consistent decoded size; "
                f"row 0 has {payload_width} bytes and row {row_index} has {len(payload)}"
            )
        decoded.append(np.frombuffer(payload, dtype=np.uint8).copy())

    assert payload_width is not None
    return np.stack(decoded, axis=0).reshape(len(decoded), payload_width, 1)


def routing_matrices_for_model_inputs(
    batch: TrainingInputBatch,
    model_input_lengths: Sequence[int],
) -> list[tuple[str, ...]]:
    """Recover exact Fireworks base64 rows from a staged SkyRL mini-batch.

    Router rows must cover every real model-input position and no other
    position.  This strict check prevents a hosted trainer from silently using
    fresh routes for an uncaptured suffix.
    """

    attention_mask = batch.get("attention_mask")
    response_mask = batch.get("response_mask")
    loss_mask = batch.get("loss_mask")
    encoded_response_rows = (batch.metadata or {}).get(
        "rollout_routing_matrices"
    )
    if encoded_response_rows is not None:
        if not isinstance(attention_mask, torch.Tensor):
            raise ValueError("Fireworks router replay requires attention_mask")
        if not isinstance(response_mask, torch.Tensor):
            raise ValueError(
                "Fireworks completion-only router replay requires response_mask"
            )
        if not isinstance(loss_mask, torch.Tensor):
            raise ValueError(
                "Fireworks completion-only router replay requires loss_mask"
            )
        if not (
            attention_mask.ndim == response_mask.ndim == loss_mask.ndim == 2
        ):
            raise ValueError(
                "Fireworks completion-only attention, response, and loss masks "
                "must be rank 2"
            )
        if len(encoded_response_rows) != len(model_input_lengths):
            raise ValueError(
                "Fireworks completion-only router replay expected one route row "
                f"per trajectory, got {len(encoded_response_rows)} for "
                f"{len(model_input_lengths)} trajectories"
            )

        encoded_batch: list[tuple[str, ...]] = []
        attention_cpu = attention_mask.detach().cpu().bool()
        response_cpu = response_mask.detach().cpu().bool()
        loss_cpu = loss_mask.detach().cpu().bool()
        for row_index, (response_routes, expected_length) in enumerate(
            zip(encoded_response_rows, model_input_lengths, strict=True)
        ):
            if expected_length <= 0:
                raise ValueError(
                    f"Fireworks router replay row {row_index} has invalid "
                    f"model-input length {expected_length}"
                )
            if not isinstance(response_routes, (list, tuple)):
                raise ValueError(
                    "Fireworks completion-only routing rows must be lists of strings"
                )
            trajectory_len = int(attention_cpu[row_index].sum().item())
            response_len = int(response_cpu[row_index].sum().item())
            prompt_len = trajectory_len - response_len
            if trajectory_len != expected_length + 1 or prompt_len <= 0:
                raise ValueError(
                    f"Fireworks router replay row {row_index} expected a one-token "
                    f"training shift, got trajectory={trajectory_len}, "
                    f"model_input={expected_length}, prompt={prompt_len}"
                )
            if len(response_routes) != response_len:
                raise ValueError(
                    "Fireworks completion-only routing rows must be response-token "
                    f"aligned: row {row_index} has {len(response_routes)} routes "
                    f"for {response_len} response tokens"
                )

            response_width = response_cpu.shape[1]
            response_slice = slice(response_width - response_len, response_width)
            trainable = loss_cpu[row_index, response_slice].tolist()
            payload_width: int | None = None
            normalized: list[str] = []
            for token_index, (encoded, is_trainable) in enumerate(
                zip(response_routes, trainable, strict=True)
            ):
                if not isinstance(encoded, str):
                    raise ValueError(
                        "Fireworks completion-only routing matrices must be strings"
                    )
                if not encoded:
                    if is_trainable:
                        raise ValueError(
                            "Fireworks completion-only router replay is missing a "
                            f"route for trainable response token {token_index} in "
                            f"row {row_index}"
                        )
                    normalized.append("")
                    continue
                try:
                    payload = base64.b64decode(encoded, validate=True)
                except (binascii.Error, ValueError) as exc:
                    raise ValueError(
                        "Fireworks completion-only routing matrix is not valid base64"
                    ) from exc
                if not payload:
                    raise ValueError(
                        "Fireworks completion-only routing matrix decoded to an "
                        "empty payload"
                    )
                if payload_width is None:
                    payload_width = len(payload)
                elif len(payload) != payload_width:
                    raise ValueError(
                        "Fireworks completion-only routing payloads must have a "
                        "consistent decoded size"
                    )
                normalized.append(encoded)

            aligned = tuple([""] * (prompt_len - 1) + normalized)
            if len(aligned) != expected_length:
                raise ValueError(
                    "Fireworks completion-only routing alignment produced "
                    f"{len(aligned)} rows for model-input length {expected_length}"
                )
            encoded_batch.append(aligned)
        return encoded_batch

    routes = batch.get("rollout_expert_indices")
    router_padding_mask = batch.get("router_padding_mask")
    if not isinstance(routes, torch.Tensor):
        raise ValueError("Fireworks router replay requires rollout_expert_indices in every staged batch")
    if not isinstance(router_padding_mask, torch.Tensor):
        raise ValueError("Fireworks router replay requires router_padding_mask in every staged batch")
    if not isinstance(attention_mask, torch.Tensor):
        raise ValueError("Fireworks router replay requires attention_mask")
    if routes.ndim != 4:
        raise ValueError(
            "Fireworks rollout_expert_indices must be rank 4 "
            f"[batch, tokens, payload, width], got {tuple(routes.shape)}"
        )
    if routes.shape[2] == 0 or routes.shape[3] == 0:
        raise ValueError("Fireworks routing payload dimensions must be non-empty")
    if router_padding_mask.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("Fireworks routing and attention masks must both be rank 2")
    if routes.shape[:2] != router_padding_mask.shape or routes.shape[:2] != attention_mask.shape:
        raise ValueError(
            "Fireworks routing tensors are not token-aligned: "
            f"routes={tuple(routes.shape)}, router_mask={tuple(router_padding_mask.shape)}, "
            f"attention_mask={tuple(attention_mask.shape)}"
        )
    if routes.shape[0] != len(model_input_lengths):
        raise ValueError(
            "Fireworks router replay expected one model-input length per row, got "
            f"{len(model_input_lengths)} for batch size {routes.shape[0]}"
        )
    if routes.dtype not in (torch.uint8, torch.int16, torch.int32):
        raise ValueError("Fireworks routing payload must use uint8, int16, or int32, " f"got {routes.dtype}")

    routes_cpu = routes.detach().cpu()
    attention_cpu = attention_mask.detach().cpu().bool()
    captured_cpu = ~router_padding_mask.detach().cpu().bool()
    encoded_batch: list[tuple[str, ...]] = []

    for row_index, expected_length in enumerate(model_input_lengths):
        if expected_length <= 0:
            raise ValueError(
                f"Fireworks router replay row {row_index} has invalid model-input length {expected_length}"
            )
        real_positions = torch.nonzero(attention_cpu[row_index], as_tuple=False).flatten()
        expected_positions = real_positions[:expected_length]
        captured_positions = torch.nonzero(captured_cpu[row_index], as_tuple=False).flatten()
        if len(real_positions) != expected_length + 1:
            raise ValueError(
                f"Fireworks router replay row {row_index} expected a one-token training shift: "
                f"{len(real_positions)} trajectory tokens for {expected_length} model-input positions"
            )
        if not torch.equal(captured_positions, expected_positions):
            raise ValueError(
                f"Fireworks router replay row {row_index} has {len(captured_positions)} aligned route rows; "
                f"expected exactly the first {expected_length} real-token positions"
            )

        selected = routes_cpu[row_index, captured_positions]
        if int(selected.min().item()) < 0 or int(selected.max().item()) > 255:
            raise ValueError(f"Fireworks routing payload row {row_index} contains values outside uint8")
        encoded_batch.append(
            tuple(
                base64.b64encode(token_row.to(dtype=torch.uint8).contiguous().numpy().tobytes()).decode("ascii")
                for token_row in selected
            )
        )

    return encoded_batch


def make_tinker_model_input(
    token_ids: Sequence[int],
    routing_matrices: Sequence[str] | None,
) -> Any:
    """Construct the SDK-patched Tinker input used by Fireworks training."""

    try:
        # Importing the Fireworks SDK installs its routing_matrices extension on
        # Tinker's ModelInput before we instantiate it.
        import fireworks.training.sdk  # noqa: F401
        import tinker
    except ImportError as exc:  # pragma: no cover - optional provider dependency
        raise ImportError("Fireworks router replay requires fireworks-ai[training] and tinker") from exc

    kwargs = {}
    if routing_matrices is not None:
        kwargs["routing_matrices"] = list(routing_matrices)
    return tinker.ModelInput.from_ints(list(token_ids), **kwargs)


def routing_payload_counts(datums: Sequence[Any]) -> tuple[int, int]:
    """Return routing-row count and decoded byte count for trainer metrics."""

    row_count = 0
    byte_count = 0
    for datum in datums:
        matrices = getattr(datum.model_input, "routing_matrices", None)
        if matrices is None:
            continue
        row_count += len(matrices)
        for encoded in matrices:
            try:
                byte_count += len(base64.b64decode(encoded, validate=True))
            except (binascii.Error, ValueError) as exc:
                raise ValueError("Trainer datum contains an invalid routing matrix") from exc
    return row_count, byte_count
