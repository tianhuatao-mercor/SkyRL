"""Version-gated ACK barrier for Fireworks chunked trainer requests."""

from __future__ import annotations

import asyncio
import inspect
import threading
import time
from dataclasses import dataclass, field
from types import MethodType
from typing import Any, Callable

EXPECTED_FIREWORKS_VERSION = "1.2.8"
EXPECTED_TINKER_VERSION = "0.23.0"


@dataclass
class ChunkAckBarrierStats:
    """Bounded evidence that each gate chunk followed all later ACKs."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    operations: int = 0
    chunks: int = 0
    multi_chunk_operations: int = 0
    minimum_gate_after_non_gate_ack_ms: float | None = None

    def record(self, *, chunks: int, gate_gap_ms: float | None) -> None:
        with self._lock:
            self.operations += 1
            self.chunks += chunks
            if chunks > 1:
                self.multi_chunk_operations += 1
            if gate_gap_ms is not None:
                current = self.minimum_gate_after_non_gate_ack_ms
                if current is None or gate_gap_ms < current:
                    self.minimum_gate_after_non_gate_ack_ms = gate_gap_ms

    def metrics(self) -> dict[str, float]:
        with self._lock:
            metrics = {
                "fireworks/chunk_ack_barrier/operations_total": float(self.operations),
                "fireworks/chunk_ack_barrier/chunks_total": float(self.chunks),
                "fireworks/chunk_ack_barrier/multi_chunk_operations_total": float(
                    self.multi_chunk_operations
                ),
            }
            if self.minimum_gate_after_non_gate_ack_ms is not None:
                metrics[
                    "fireworks/chunk_ack_barrier/minimum_gate_after_non_gate_ack_ms"
                ] = self.minimum_gate_after_non_gate_ack_ms
            return metrics


def install_chunk_ack_barrier(client: Any) -> ChunkAckBarrierStats:
    """Replace one Fireworks client sender with ACK-before-gate admission."""

    import fireworks
    import tinker
    from tinker.lib.api_future_impl import _APIFuture, _CombinedAPIFuture
    from tinker.lib.chunked_fwdbwd_helpers import combine_fwd_bwd_output_results

    if getattr(fireworks, "__version__", None) != EXPECTED_FIREWORKS_VERSION:
        raise RuntimeError(
            "chunk ACK barrier audited only for fireworks-ai "
            f"{EXPECTED_FIREWORKS_VERSION}"
        )
    if getattr(tinker, "__version__", None) != EXPECTED_TINKER_VERSION:
        raise RuntimeError(
            f"chunk ACK barrier audited only for tinker {EXPECTED_TINKER_VERSION}"
        )
    if getattr(client, "_skyrl_chunk_ack_barrier_installed", False):
        raise RuntimeError("chunk ACK barrier is already installed on this client")

    original = getattr(client, "_run_chunked_requests", None)
    if original is None:
        raise RuntimeError("training client has no _run_chunked_requests method")
    signature = inspect.signature(original)
    if tuple(signature.parameters) != ("requests", "send_chunk", "request_type"):
        raise RuntimeError(f"unexpected _run_chunked_requests signature: {signature}")

    stats = ChunkAckBarrierStats()

    async def run_with_barrier(
        self: Any,
        requests: list[tuple[int, list[Any]]],
        send_chunk: Callable[..., Any],
        *,
        request_type: str,
    ) -> Any:
        if not requests:
            raise ValueError("No data provided")
        parallel = self._parallel_chunks_enabled()
        min_rid = requests[0][0]
        max_rid = requests[-1][0] if parallel else None
        request_started = time.time()
        non_gate_ack_ns: list[int] = []

        async def submit(request_id: int, chunk: list[Any], *, gate: bool) -> Any:
            turn_min = min_rid if parallel else request_id
            turn_max = max_rid if parallel else None
            async with self._take_turn(turn_min, turn_max):
                untyped = await self.holder.execute_with_retries(
                    send_chunk,
                    request_id,
                    chunk,
                )
            if not gate:
                non_gate_ack_ns.append(time.monotonic_ns())
            return _APIFuture(
                tinker.types.ForwardBackwardOutput,
                self.holder,
                untyped,
                request_start_time=request_started,
                request_type=request_type,
                queue_state_observer=self._queue_state_logger,
            )

        gate_gap_ms: float | None = None
        if parallel and len(requests) > 1:
            rest = list(
                await asyncio.gather(
                    *[
                        submit(request_id, chunk, gate=False)
                        for request_id, chunk in requests[1:]
                    ]
                )
            )
            gate_submit_ns = time.monotonic_ns()
            if not non_gate_ack_ns:
                raise RuntimeError("chunk ACK barrier observed no non-gate ACKs")
            gate_gap_ms = (gate_submit_ns - max(non_gate_ack_ns)) / 1_000_000
            if gate_gap_ms < 0:
                raise RuntimeError("gate chunk was submitted before a non-gate ACK")
            first_id, first_chunk = requests[0]
            first = await submit(first_id, first_chunk, gate=True)
            futures = [first] + rest
        else:
            futures = list(
                await asyncio.gather(
                    *[
                        submit(request_id, chunk, gate=(index == 0))
                        for index, (request_id, chunk) in enumerate(requests)
                    ]
                )
            )
        stats.record(chunks=len(requests), gate_gap_ms=gate_gap_ms)
        return _CombinedAPIFuture(
            futures,
            combine_fwd_bwd_output_results,
            self.holder,
        )

    client._run_chunked_requests = MethodType(run_with_barrier, client)
    client._skyrl_chunk_ack_barrier_installed = True
    return stats
