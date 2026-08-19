import asyncio
import sys
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from skyrl.backends.fireworks.chunk_ack import install_chunk_ack_barrier


class _Combined:
    def __init__(self, futures, combine, holder):
        self.futures = futures
        self.combine = combine
        self.holder = holder


class _Future:
    def __init__(self, output_type, holder, untyped, **kwargs):
        self.output_type = output_type
        self.holder = holder
        self.untyped = untyped
        self.kwargs = kwargs


class _Holder:
    def __init__(self, events):
        self.events = events

    async def execute_with_retries(self, send_chunk, request_id, chunk):
        self.events.append(f"send:{request_id}")
        value = await send_chunk(request_id, chunk)
        self.events.append(f"ack:{request_id}")
        return value


class _Client:
    def __init__(self, events):
        self.events = events
        self.holder = _Holder(events)
        self._queue_state_logger = object()

    def _parallel_chunks_enabled(self):
        return True

    @asynccontextmanager
    async def _take_turn(self, minimum, maximum):
        del minimum, maximum
        yield

    async def _run_chunked_requests(self, requests, send_chunk, *, request_type):
        del requests, send_chunk, request_type
        raise AssertionError("must be replaced")


def test_chunk_ack_barrier_acks_later_chunks_before_gate(monkeypatch) -> None:
    import fireworks
    import tinker
    import tinker.lib.api_future_impl as future_impl
    import tinker.lib.chunked_fwdbwd_helpers as helpers

    monkeypatch.setattr(fireworks, "__version__", "1.2.8")
    monkeypatch.setattr(tinker, "__version__", "0.23.0")
    monkeypatch.setattr(future_impl, "_APIFuture", _Future)
    monkeypatch.setattr(future_impl, "_CombinedAPIFuture", _Combined)
    monkeypatch.setattr(helpers, "combine_fwd_bwd_output_results", object())
    monkeypatch.setattr(
        tinker,
        "types",
        SimpleNamespace(ForwardBackwardOutput=object()),
    )
    events = []
    client = _Client(events)
    gate_seen_after = []

    async def send(request_id, chunk):
        del chunk
        if request_id == 0:
            gate_seen_after.extend(events)
        await asyncio.sleep(0)
        return SimpleNamespace(request_id=f"provider-{request_id}")

    stats = install_chunk_ack_barrier(client)
    combined = asyncio.run(
        client._run_chunked_requests(
            [(0, ["gate"]), (1, ["rest-1"]), (2, ["rest-2"])],
            send,
            request_type="ForwardBackward",
        )
    )

    assert isinstance(combined, _Combined)
    assert "ack:1" in gate_seen_after
    assert "ack:2" in gate_seen_after
    metrics = stats.metrics()
    assert metrics["fireworks/chunk_ack_barrier/operations_total"] == 1.0
    assert metrics["fireworks/chunk_ack_barrier/chunks_total"] == 3.0
    assert metrics[
        "fireworks/chunk_ack_barrier/minimum_gate_after_non_gate_ack_ms"
    ] >= 0.0


def test_chunk_ack_barrier_fails_closed_on_version(monkeypatch) -> None:
    import fireworks
    import tinker

    monkeypatch.setattr(fireworks, "__version__", "new-version")
    monkeypatch.setattr(tinker, "__version__", "0.23.0")

    with pytest.raises(RuntimeError, match="audited only"):
        install_chunk_ack_barrier(_Client([]))
