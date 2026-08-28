"""Length-aware sampler for balanced packed SFT updates.

This sampler is intended for the common packed-SFT layout where a global
batch contains exactly two source trajectories per data-parallel rank.  It
pairs the globally longest and shortest trajectories, sorts those pairs by an
additive transformer-work proxy, and groups adjacent pairs into global
batches.  Batch order is shuffled deterministically each epoch.

The result keeps each trajectory intact and uses every trajectory exactly
once per epoch.  It changes only sample order; the packed collator remains
responsible for block-diagonal attention and loss-mask construction.

Example::

    sampler=custom \
    sampler_class_path=examples.train.sft.pairwise_flops_sampler.PairwiseFlopsBatchSampler \
    'sampler_kwargs={batch_size: 64, quadratic_equivalent_length: 262196, max_tokens_per_microbatch: 262144, seed: 42}'

``quadratic_equivalent_length`` expresses the ratio between the model's
linear and quadratic FLOP coefficients.  The ordering proxy is proportional
to ``length * equivalent_length + length**2`` and therefore requires no
floating-point arithmetic.
"""

from __future__ import annotations

import random
from typing import Iterator, Sequence

import torch
from loguru import logger


class PairwiseFlopsBatchSampler(torch.utils.data.Sampler[int]):
    """Emit deterministic, packed-work-balanced global batches.

    Args:
        data_source: Dataset exposing an integer ``sequence_lengths`` entry per
            sample.
        batch_size: Number of source trajectories in one global optimizer
            update.  It must be even and the dataset size must be divisible by
            it.  The intended layout is ``batch_size == 2 * dp_size``.
        quadratic_equivalent_length: Positive integer ``linear_flops /
            quadratic_flops`` for the model.  Sequence work is ordered by
            ``L * quadratic_equivalent_length + L**2``.
        max_tokens_per_microbatch: Optional packed-bin capacity.  When set,
            construction fails if a longest/shortest pair would exceed the
            capacity rather than silently creating extra packed bins.
        seed: Seed used only to shuffle the order of already-balanced global
            batches.  Every epoch uses a distinct deterministic shuffle.
    """

    def __init__(
        self,
        data_source: Sequence,
        *,
        batch_size: int,
        quadratic_equivalent_length: int,
        max_tokens_per_microbatch: int | None = None,
        seed: int = 0,
    ) -> None:
        self.data_source = data_source
        self.batch_size = int(batch_size)
        self.quadratic_equivalent_length = int(quadratic_equivalent_length)
        self.max_tokens_per_microbatch = (
            None if max_tokens_per_microbatch is None else int(max_tokens_per_microbatch)
        )
        self.seed = int(seed)
        self.position = 0
        self.epoch = 0
        self._plan: list[int] | None = None

        n = len(data_source)
        if self.batch_size < 2 or self.batch_size % 2:
            raise ValueError(f"PairwiseFlopsBatchSampler: batch_size must be an even integer >= 2, got {batch_size}.")
        if n == 0 or n % self.batch_size:
            raise ValueError(
                "PairwiseFlopsBatchSampler: dataset size must be positive and divisible by "
                f"batch_size; got len(data_source)={n}, batch_size={self.batch_size}."
            )
        if self.quadratic_equivalent_length <= 0:
            raise ValueError(
                "PairwiseFlopsBatchSampler: quadratic_equivalent_length must be positive, "
                f"got {quadratic_equivalent_length}."
            )
        if self.max_tokens_per_microbatch is not None and self.max_tokens_per_microbatch <= 0:
            raise ValueError(
                "PairwiseFlopsBatchSampler: max_tokens_per_microbatch must be positive when set, "
                f"got {max_tokens_per_microbatch}."
            )

        try:
            lengths = data_source.sequence_lengths
        except AttributeError as exc:
            raise ValueError(
                "PairwiseFlopsBatchSampler: data_source must expose sequence_lengths."
            ) from exc
        self.sequence_lengths = [int(length) for length in lengths]
        if len(self.sequence_lengths) != n:
            raise ValueError(
                "PairwiseFlopsBatchSampler: sequence_lengths must align with data_source; "
                f"got {len(self.sequence_lengths)} lengths for {n} samples."
            )
        if any(length <= 0 for length in self.sequence_lengths):
            raise ValueError("PairwiseFlopsBatchSampler: all sequence lengths must be positive.")

        # Validate the exact per-batch endpoint pairing that the fixed-bin LPT
        # collator will reconstruct.  This proves the one-bin-per-pair capacity
        # invariant for the whole run, including cross-pair regrouping.
        pairs = self._build_pairs()
        batches = self._build_balanced_batches(pairs)
        batch_pair_costs = [self._packed_pair_costs(batch) for batch in batches]
        pair_costs = [self._pair_cost(pair) for pair in pairs]
        max_batch_imbalance = max(max(costs) / min(costs) for costs in batch_pair_costs)
        logger.info(
            "PairwiseFlopsBatchSampler prepared {} trajectories as {} pairs / {} balanced batches; "
            "pair-work min={} mean={:.1f} max={}, max_in_batch_imbalance={:.4f}, seed={}",
            n,
            len(pairs),
            n // self.batch_size,
            min(pair_costs),
            sum(pair_costs) / len(pair_costs),
            max(pair_costs),
            max_batch_imbalance,
            self.seed,
        )

    def _sequence_cost(self, idx: int) -> int:
        length = self.sequence_lengths[idx]
        return length * self.quadratic_equivalent_length + length * length

    def _pair_cost(self, pair: tuple[int, int]) -> int:
        return self._sequence_cost(pair[0]) + self._sequence_cost(pair[1])

    def _build_pairs(self) -> list[tuple[int, int]]:
        ordered = sorted(range(len(self.sequence_lengths)), key=lambda idx: (self.sequence_lengths[idx], idx))
        half = len(ordered) // 2
        return [(ordered[i], ordered[-1 - i]) for i in range(half)]

    def _packed_pairs(self, batch: list[int]) -> list[tuple[int, int]]:
        ordered = sorted(batch, key=lambda idx: (self.sequence_lengths[idx], idx))
        half = len(ordered) // 2
        return [(ordered[i], ordered[-1 - i]) for i in range(half)]

    def _packed_pair_costs(self, batch: list[int]) -> list[int]:
        packed_pairs = self._packed_pairs(batch)
        if self.max_tokens_per_microbatch is not None:
            for left, right in packed_pairs:
                pair_tokens = self.sequence_lengths[left] + self.sequence_lengths[right]
                if pair_tokens > self.max_tokens_per_microbatch:
                    raise ValueError(
                        "PairwiseFlopsBatchSampler: balanced batch exceeds max_tokens_per_microbatch; "
                        f"{self.sequence_lengths[left]} + {self.sequence_lengths[right]} = {pair_tokens} > "
                        f"{self.max_tokens_per_microbatch}."
                    )
        return [self._pair_cost(pair) for pair in packed_pairs]

    def _build_balanced_batches(self, pairs: list[tuple[int, int]]) -> list[list[int]]:
        pairs = sorted(pairs, key=lambda pair: (self._pair_cost(pair), pair))
        pairs_per_batch = self.batch_size // 2
        batches: list[list[int]] = []
        for start in range(0, len(pairs), pairs_per_batch):
            batch_pairs = pairs[start : start + pairs_per_batch]
            batch = [idx for pair in batch_pairs for idx in pair]
            if len(batch) != self.batch_size:
                raise AssertionError("balanced sampler constructed a partial batch")
            batches.append(batch)
        return batches

    def _build_plan(self) -> list[int]:
        batches = self._build_balanced_batches(self._build_pairs())
        rng = random.Random(self.seed + self.epoch)
        rng.shuffle(batches)
        return [idx for batch in batches for idx in batch]

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
        if position < 0 or position > len(self.data_source):
            raise ValueError(f"PairwiseFlopsBatchSampler: invalid restored position {position}.")
        if epoch < 0:
            raise ValueError(f"PairwiseFlopsBatchSampler: invalid restored epoch {epoch}.")
        self.position = position
        self.epoch = epoch
        self._plan = None
