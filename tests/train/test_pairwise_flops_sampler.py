"""Tests for the packed-SFT pairwise FLOP-balanced sampler example."""

from __future__ import annotations

from examples.train.sft.pairwise_flops_sampler import PairwiseFlopsBatchSampler


class _LengthDataset:
    def __init__(self, lengths):
        self.sequence_lengths = list(lengths)

    def __len__(self):
        return len(self.sequence_lengths)


def _batches(values, batch_size):
    return [values[start : start + batch_size] for start in range(0, len(values), batch_size)]


def _cost(length, equivalent):
    return length * equivalent + length * length


def test_each_epoch_is_a_complete_permutation_and_batches_fit():
    lengths = [10, 12, 15, 18, 22, 25, 30, 32, 70, 74, 78, 82, 86, 90, 94, 98]
    dataset = _LengthDataset(lengths)
    sampler = PairwiseFlopsBatchSampler(
        dataset,
        batch_size=8,
        quadratic_equivalent_length=100,
        max_tokens_per_microbatch=110,
        seed=7,
    )

    epoch = list(sampler)
    assert sorted(epoch) == list(range(len(lengths)))
    for batch in _batches(epoch, 8):
        ordered = sorted((lengths[idx] for idx in batch))
        paired = zip(ordered, reversed(ordered))
        assert all(left + right <= 110 for left, right in list(paired)[:4])


def test_batches_have_tightly_grouped_pair_work():
    lengths = list(range(10, 42))
    equivalent = 100
    sampler = PairwiseFlopsBatchSampler(
        _LengthDataset(lengths),
        batch_size=8,
        quadratic_equivalent_length=equivalent,
        seed=3,
    )

    for batch in _batches(list(sampler), 8):
        ordered = sorted(batch, key=lambda idx: lengths[idx])
        pair_costs = [
            _cost(lengths[ordered[i]], equivalent) + _cost(lengths[ordered[-1 - i]], equivalent)
            for i in range(4)
        ]
        assert max(pair_costs) / min(pair_costs) < 1.03


def test_seed_is_deterministic_and_only_changes_batch_order():
    dataset = _LengthDataset(range(1, 33))
    kwargs = dict(batch_size=8, quadratic_equivalent_length=50)
    plan_a = list(PairwiseFlopsBatchSampler(dataset, seed=11, **kwargs))
    plan_b = list(PairwiseFlopsBatchSampler(dataset, seed=11, **kwargs))
    plan_c = list(PairwiseFlopsBatchSampler(dataset, seed=12, **kwargs))
    assert plan_a == plan_b
    assert plan_a != plan_c
    assert sorted(map(sorted, _batches(plan_a, 8))) == sorted(map(sorted, _batches(plan_c, 8)))


def test_resume_reconstructs_the_exact_remaining_plan():
    dataset = _LengthDataset(range(1, 33))
    kwargs = dict(batch_size=8, quadratic_equivalent_length=50, seed=4)
    sampler = PairwiseFlopsBatchSampler(dataset, **kwargs)
    iterator = iter(sampler)
    consumed = [next(iterator) for _ in range(13)]
    state = sampler.state_dict()
    expected_rest = list(iterator)

    resumed = PairwiseFlopsBatchSampler(dataset, **kwargs)
    resumed.load_state_dict(state)
    assert list(resumed) == expected_rest
    assert sorted(consumed + expected_rest) == list(range(32))


def test_next_epoch_uses_a_distinct_deterministic_batch_order():
    dataset = _LengthDataset(range(1, 33))
    kwargs = dict(batch_size=8, quadratic_equivalent_length=50, seed=8)
    sampler = PairwiseFlopsBatchSampler(dataset, **kwargs)
    first = list(sampler)
    second = list(sampler)
    replay = PairwiseFlopsBatchSampler(dataset, **kwargs)
    assert first == list(replay)
    assert second == list(replay)
    assert first != second
