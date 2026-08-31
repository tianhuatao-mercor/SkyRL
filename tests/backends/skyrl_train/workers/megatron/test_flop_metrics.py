"""Unit coverage for corrections around Megatron Bridge FLOP estimates."""

from types import SimpleNamespace

import pytest

from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    _hybrid_packed_attention_flops_correction,
)


def _provider(**kwargs):
    defaults = {
        "is_hybrid_model": True,
        "hybrid_layer_pattern": None,
        "num_layers": 8,
        "hybrid_attention_ratio": 0.25,
        "kv_channels": 256,
        "num_attention_heads": 24,
        "hidden_size": 5120,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_hybrid_attention_correction_replaces_average_square():
    # Two full-attention layers, Q projection size 256*24. The two sequences
    # have lengths 3 and 7, so exact sum-of-squares exceeds the average-length
    # estimate by (9+49) - 2*5**2 = 8.
    correction = _hybrid_packed_attention_flops_correction(
        _provider(),
        batch_size=2,
        seqlen_sum=10,
        seqlen_squared_sum=58,
    )

    assert correction == 6 * 2 * (256 * 24) * 8


def test_hybrid_attention_correction_is_zero_for_equal_lengths():
    assert (
        _hybrid_packed_attention_flops_correction(
            _provider(),
            batch_size=2,
            seqlen_sum=10,
            seqlen_squared_sum=50,
        )
        == 0
    )


def test_hybrid_attention_correction_rejects_inconsistent_stats():
    with pytest.raises(ValueError, match="inconsistent"):
        _hybrid_packed_attention_flops_correction(
            _provider(),
            batch_size=2,
            seqlen_sum=10,
            seqlen_squared_sum=49,
        )
