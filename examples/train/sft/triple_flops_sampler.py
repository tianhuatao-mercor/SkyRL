"""Deterministic three-trajectories-per-DP-rank sampler for packed SFT."""

from __future__ import annotations

import random
from typing import Iterator, Sequence

import torch
from loguru import logger


class TripleFlopsBatchSampler(torch.utils.data.Sampler[int]):
    """Build FLOP-balanced triples, then adjacent-cost global batches.

    The intended layout is ``batch_size == 3 * dp_size`` together with the
    ``fixed_bin_flops_balanced`` collator. Every trajectory remains intact and
    appears exactly once per epoch; only deterministic sample order changes.
    """

    def __init__(
        self,
        data_source: Sequence,
        *,
        batch_size: int,
        quadratic_equivalent_length: int,
        max_tokens_per_microbatch: int,
        seed: int = 0,
    ) -> None:
        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.quadratic_equivalent_length = int(quadratic_equivalent_length)
        self.max_tokens_per_microbatch = int(max_tokens_per_microbatch)
        self.seed = int(seed)
        self.position = 0
        self.epoch = 0
        self._plan: list[int] | None = None

        n = len(data_source)
        if self.batch_size < 3 or self.batch_size % 3:
            raise ValueError("TripleFlopsBatchSampler: batch_size must be a multiple of 3.")
        if n == 0 or n % self.batch_size:
            raise ValueError(
                "TripleFlopsBatchSampler: dataset size must be positive and divisible by "
                f"batch_size; got len(data_source)={n}, batch_size={self.batch_size}."
            )
        if self.quadratic_equivalent_length <= 0 or self.max_tokens_per_microbatch <= 0:
            raise ValueError("TripleFlopsBatchSampler: FLOP crossover and token capacity must be positive.")

        try:
            lengths = data_source.sequence_lengths
        except AttributeError as exc:
            raise ValueError("TripleFlopsBatchSampler: data_source must expose sequence_lengths.") from exc
        self.sequence_lengths = [int(length) for length in lengths]
        if len(self.sequence_lengths) != n or any(length <= 0 for length in self.sequence_lengths):
            raise ValueError("TripleFlopsBatchSampler: sequence_lengths must be positive and align with data_source.")

        groups = self._build_groups()
        batches = self._build_batches(groups)
        group_costs = [self._group_cost(group) for group in groups]
        max_batch_imbalance = max(
            max(self._group_cost(group) for group in batch)
            / min(self._group_cost(group) for group in batch)
            for batch in batches
        )
        logger.info(
            "TripleFlopsBatchSampler prepared {} trajectories as {} triples / {} balanced batches; "
            "triple-work min={} mean={:.1f} max={}, max_in_batch_imbalance={:.4f}, seed={}",
            n,
            len(groups),
            len(batches),
            min(group_costs),
            sum(group_costs) / len(group_costs),
            max(group_costs),
            max_batch_imbalance,
            self.seed,
        )

    def _sequence_cost(self, idx: int) -> int:
        length = self.sequence_lengths[idx]
        return length * self.quadratic_equivalent_length + length * length

    def _group_cost(self, group: list[int]) -> int:
        return sum(self._sequence_cost(idx) for idx in group)

    def _build_groups(self) -> list[list[int]]:
        ordered = sorted(
            range(len(self.sequence_lengths)),
            key=lambda idx: (self._sequence_cost(idx), idx),
            reverse=True,
        )
        num_groups = len(ordered) // 3
        groups = [[idx] for idx in ordered[:num_groups]]
        group_work = [self._sequence_cost(idx) for idx in ordered[:num_groups]]
        group_tokens = [self.sequence_lengths[idx] for idx in ordered[:num_groups]]

        for idx in ordered[num_groups:]:
            length = self.sequence_lengths[idx]
            candidates = [
                group_idx
                for group_idx, group in enumerate(groups)
                if len(group) < 3 and group_tokens[group_idx] + length <= self.max_tokens_per_microbatch
            ]
            if not candidates:
                raise ValueError(
                    "TripleFlopsBatchSampler: no capacity-preserving triple assignment exists for "
                    f"sequence length {length}."
                )
            group_idx = min(candidates, key=lambda i: (group_work[i], group_tokens[i], i))
            groups[group_idx].append(idx)
            group_work[group_idx] += self._sequence_cost(idx)
            group_tokens[group_idx] += length

        if not all(len(group) == 3 for group in groups):
            raise AssertionError("TripleFlopsBatchSampler constructed an incomplete triple.")
        return groups

    def _build_batches(self, groups: list[list[int]]) -> list[list[list[int]]]:
        ordered = sorted(groups, key=lambda group: (self._group_cost(group), group))
        groups_per_batch = self.batch_size // 3
        batches = [ordered[start : start + groups_per_batch] for start in range(0, len(ordered), groups_per_batch)]
        if not all(len(batch) == groups_per_batch for batch in batches):
            raise AssertionError("TripleFlopsBatchSampler constructed a partial batch.")
        return batches

    def _build_plan(self) -> list[int]:
        batches = self._build_batches(self._build_groups())
        random.Random(self.seed + self.epoch).shuffle(batches)
        return [idx for batch in batches for group in batch for idx in group]

    def __iter__(self) -> Iterator[int]:
        if self._plan is None:
            self._plan = self._build_plan()
        while self.position < len(self._plan):
            idx = self._plan[self.position]
            self.position += 1
            yield idx
        self.position = 0
        self.epoch += 1
        self._plan = None

    def __len__(self) -> int:
        return len(self.data_source)

    def state_dict(self) -> dict[str, int]:
        return {"position": self.position, "epoch": self.epoch}

    def load_state_dict(self, state: dict[str, int]) -> None:
        position = int(state["position"])
        epoch = int(state.get("epoch", 0))
        if position < 0 or position > len(self.data_source) or epoch < 0:
            raise ValueError("TripleFlopsBatchSampler: invalid restored sampler state.")
        self.position = position
        self.epoch = epoch
        self._plan = None
