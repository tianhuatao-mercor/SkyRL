import pytest

from skyrl.backends.skyrl_train.workers.megatron.megatron_worker import (
    _maybe_override_fla_causal_conv_backend,
)


def test_fla_causal_conv_backend_unset_is_noop(monkeypatch):
    monkeypatch.delenv("SKYRL_FLA_CAUSAL_CONV_BACKEND", raising=False)
    assert _maybe_override_fla_causal_conv_backend() is None


def test_fla_causal_conv_backend_rejects_unknown_value(monkeypatch):
    monkeypatch.setenv("SKYRL_FLA_CAUSAL_CONV_BACKEND", "unknown")
    with pytest.raises(ValueError, match="must be one of"):
        _maybe_override_fla_causal_conv_backend()


def test_fla_causal_conv_backend_wraps_megatron_call(monkeypatch):
    from megatron.core.ssm import gated_delta_net

    calls = []

    def stub(*args, **kwargs):
        calls.append((args, kwargs))
        return "result"

    monkeypatch.setattr(gated_delta_net, "causal_conv1d", stub)
    monkeypatch.setenv("SKYRL_FLA_CAUSAL_CONV_BACKEND", "cuda")

    assert _maybe_override_fla_causal_conv_backend() == "cuda"
    assert gated_delta_net.causal_conv1d("x", activation="silu") == "result"
    assert calls == [(('x',), {"activation": "silu", "backend": "cuda"})]
    assert _maybe_override_fla_causal_conv_backend() == "cuda"
