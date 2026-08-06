"""Lifecycle for a dedicated Fireworks trainer and rollout deployment."""

from __future__ import annotations

import asyncio
import os
import re
import threading
import time
import uuid
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from skyrl.backends.tinker.runtime import configure_tinker_pyqwest_system_certs
from skyrl.train.config import FireworksConfig


@dataclass(frozen=True)
class SamplerVersion:
    """Snapshot identity visible to rollout calls admitted after publication."""

    version: int
    snapshot_path: str


@dataclass(frozen=True)
class FireworksInferenceEndpoint:
    """Native OpenAI-compatible endpoint for the managed deployment."""

    api_base: str
    model: str


def _close_quietly(resource: Any) -> None:
    close = getattr(resource, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:  # pragma: no cover - provider-specific teardown
        warnings.warn(
            "Failed to close Fireworks resource "
            f"{type(resource).__name__}; exception details omitted because "
            "provider errors may contain credential-bearing headers.",
            RuntimeWarning,
        )


class FireworksRuntime:
    """Own one dedicated trainer, deployment, and stable sampling client.

    Fireworks hot-loads new snapshots into the deployment without replacing
    its URL or the local sampling client. ``_active_samples`` exists only so
    teardown does not close that client underneath an in-flight request; it
    does not represent multiple model clients or deployed versions.
    """

    _RESTORABLE_USAGE_METRICS = {
        "fireworks/usage/sampling_requests_total": "_sampling_requests",
        "fireworks/usage/prompt_tokens_total": "_prompt_tokens",
        "fireworks/usage/prompt_cache_hit_tokens_total": "_prompt_cache_hit_tokens",
        "fireworks/usage/prompt_cache_unknown_tokens_total": (
            "_prompt_cache_unknown_tokens"
        ),
        "fireworks/usage/sampled_tokens_total": "_sampled_tokens",
        "fireworks/usage/sampling_request_seconds_total": "_sampling_request_seconds",
        "fireworks/usage/forward_backward_calls_total": "_forward_backward_calls",
        "fireworks/usage/training_tokens_total": "_training_tokens",
        "fireworks/usage/forward_backward_seconds_total": "_forward_backward_seconds",
    }

    def __init__(
        self,
        *,
        service: Any,
        training_client: Any,
        tokenizer: Any,
        config: FireworksConfig,
        started_monotonic: float | None = None,
        started_at_utc: str | None = None,
    ) -> None:
        self.service = service
        self.training_client = training_client
        self.tokenizer = tokenizer
        self.config = config
        self._state_lock = threading.Condition()
        self._publish_lock = asyncio.Lock()
        self._sampler: Any | None = None
        self._sampler_identity: SamplerVersion | None = None
        self._active_samples = 0
        self._next_version = 0
        self._closed = False
        self._started_monotonic = (
            time.monotonic() if started_monotonic is None else started_monotonic
        )
        self._started_at_utc = started_at_utc or datetime.now(UTC).isoformat()
        self._current_stage_started_at_utc = self._started_at_utc
        self._usage_lock = threading.Lock()
        self._restored_wall_time_seconds = 0.0
        self._restored_trainer_gpu_hours = 0.0
        self._restored_rollout_gpu_hours = 0.0
        self._restored_gpu_cost_usd = 0.0
        self._restored_billing_stages: list[dict[str, Any]] = []
        self._cost_history_complete = True
        self._usage_restored = False
        self._sampling_requests = 0
        self._prompt_tokens = 0
        self._prompt_cache_hit_tokens = 0
        self._prompt_cache_unknown_tokens = 0
        self._sampled_tokens = 0
        self._sampling_request_seconds = 0.0
        self._forward_backward_calls = 0
        self._training_tokens = 0
        self._forward_backward_seconds = 0.0

    @classmethod
    def connect(
        cls,
        *,
        config: FireworksConfig,
        tokenizer: Any,
        tokenizer_model: str,
        lora_rank: int,
        lora_alpha: int,
        learning_rate: float,
    ) -> FireworksRuntime:
        """Provision an SDK-managed trainer and linked rollout deployment."""

        started_monotonic = time.monotonic()
        started_at_utc = datetime.now(UTC).isoformat()
        api_key = os.environ.get("FIREWORKS_API_KEY")
        if not api_key:
            raise RuntimeError("FIREWORKS_API_KEY is required for Fireworks training")
        if not config.base_model or not config.training_shape_id:
            raise ValueError(
                "Dedicated Fireworks training requires base_model and training_shape_id"
            )
        if not config.trainer_job_id or not config.deployment_id:
            raise ValueError(
                "Dedicated Fireworks training requires stable trainer_job_id and deployment_id"
            )

        try:
            from fireworks.training.sdk import FiretitanServiceClient
        except ImportError as exc:
            raise ImportError(
                "The Fireworks backend requires fireworks-ai[training]; "
                "install SkyRL with --extra fireworks"
            ) from exc

        # Fireworks' embedded Tinker client enables pyqwest. On macOS, its
        # default CA bundle can omit trusted enterprise/system roots, so keep
        # TLS verification enabled while adding the operating-system CA store.
        configure_tinker_pyqwest_system_certs()

        cleanup_deployment = (
            config.cleanup_deployment_on_close if config.cleanup_on_exit else None
        )
        service = FiretitanServiceClient.from_firetitan_config(
            api_key=api_key,
            base_url=config.base_url,
            base_model=config.base_model,
            tokenizer_model=tokenizer_model,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            training_shape_id=config.training_shape_id,
            trainer_job_id=config.trainer_job_id,
            deployment_id=config.deployment_id,
            create_deployment=True,
            max_context_length=config.max_seq_len,
            learning_rate=learning_rate,
            # SkyRL performs gradient accumulation explicitly by issuing
            # forward_backward calls before optim_step. The Fireworks SDK's
            # legacy server-side field is deprecated, so omit it.
            gradient_accumulation_steps=None,
            trainer_replica_count=config.trainer_replica_count,
            replica_count=config.replica_count,
            trainer_timeout_s=config.trainer_timeout_s,
            deployment_timeout_s=config.deployment_timeout_s,
            hotload_timeout_s=config.hotload_timeout_s,
            cleanup_trainer_on_close=config.cleanup_on_exit,
            cleanup_deployment_on_close=cleanup_deployment,
        )
        try:
            training_client = service.create_training_client(
                base_model=config.base_model,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha,
            )
        except BaseException:
            # Include KeyboardInterrupt so a wall-clock supervisor can still
            # invoke SDK cleanup during a long provisioning wait.
            _close_quietly(service)
            raise
        return cls(
            service=service,
            training_client=training_client,
            tokenizer=tokenizer,
            config=config,
            started_monotonic=started_monotonic,
            started_at_utc=started_at_utc,
        )

    @property
    def trainer_job_id(self) -> str | None:
        return getattr(self.service, "trainer_job_id", None)

    @property
    def deployment_id(self) -> str | None:
        return getattr(self.service, "deployment_id", None)

    @property
    def inference_endpoint(self) -> FireworksInferenceEndpoint:
        """Return the native endpoint backing the managed rollout deployment."""

        handle = getattr(self.service, "_managed_handle", None)
        deployment = getattr(handle, "deployment", None)
        deployment_manager = getattr(handle, "deployment_manager", None)
        model = getattr(deployment, "inference_model", None)
        inference_url = getattr(deployment_manager, "inference_url", None)
        if not model or not inference_url:
            raise RuntimeError(
                "Fireworks SDK-managed deployment did not expose its native "
                "inference model and URL after provisioning"
            )
        return FireworksInferenceEndpoint(
            api_base=f"{str(inference_url).rstrip('/')}/inference/v1",
            model=str(model),
        )

    @property
    def weight_version(self) -> int:
        with self._state_lock:
            if self._sampler_identity is None:
                return -1
            return self._sampler_identity.version

    @property
    def snapshot_path(self) -> str | None:
        with self._state_lock:
            if self._sampler_identity is None:
                return None
            return self._sampler_identity.snapshot_path

    def restore_usage_reports(self, reports: list[dict[str, Any]]) -> None:
        """Restore cumulative usage without repricing earlier run segments.

        A resumed job may use a different trainer or rollout replica count.
        Historical GPU-hours and dollars therefore remain fixed at the
        topology and rate recorded by the checkpoint, while the current
        process accrues a new billing segment from its own configuration.
        """

        if (
            len(reports) != 1
            or reports[0].get("cumulative_across_resumes") is not True
        ):
            raise RuntimeError(
                "Fireworks resume requires one cumulative current-format usage report"
            )

        report = reports[0]
        metrics = report.get("metrics")
        if not isinstance(metrics, dict):
            raise RuntimeError("Fireworks checkpoint usage metrics must be an object")
        restored_wall_time = float(
            metrics.get("fireworks/usage/wall_time_seconds", 0.0)
        )
        if restored_wall_time < 0:
            raise RuntimeError("Fireworks checkpoint usage contains negative wall time")
        restored_billing = self._billing_totals_from_report(report)
        saved_stages = report.get("billing_stages")
        if not isinstance(saved_stages, list):
            raise RuntimeError(
                "Fireworks checkpoint usage is missing billing_stages; "
                "only the current accounting format is supported"
            )
        if not all(isinstance(stage, dict) for stage in saved_stages):
            raise RuntimeError(
                "Fireworks checkpoint billing_stages must contain objects"
            )

        with self._usage_lock:
            if self._usage_restored:
                raise RuntimeError("Fireworks usage has already been restored")
            if any(
                (
                    self._sampling_requests,
                    self._forward_backward_calls,
                    self._training_tokens,
                )
            ):
                raise RuntimeError(
                    "Fireworks usage must be restored before recording provider work"
                )
            self._restored_wall_time_seconds += restored_wall_time
            for metric_name, attribute_name in self._RESTORABLE_USAGE_METRICS.items():
                value = metrics.get(metric_name, 0)
                setattr(
                    self,
                    attribute_name,
                    getattr(self, attribute_name) + value,
                )
            if restored_billing is None:
                self._cost_history_complete = False
            else:
                trainer_gpu_hours, rollout_gpu_hours, gpu_cost_usd = restored_billing
                self._restored_trainer_gpu_hours += trainer_gpu_hours
                self._restored_rollout_gpu_hours += rollout_gpu_hours
                self._restored_gpu_cost_usd += gpu_cost_usd
            self._restored_billing_stages.extend(dict(stage) for stage in saved_stages)
            started_at = report.get("started_at_utc")
            if started_at:
                self._started_at_utc = min(self._started_at_utc, str(started_at))
            self._usage_restored = True

    @staticmethod
    def _nonnegative_float(value: Any) -> float | None:
        if value is None:
            return None
        number = float(value)
        if number < 0:
            raise RuntimeError(
                "Fireworks checkpoint usage contains a negative billing value"
            )
        return number

    def _billing_totals_from_report(
        self,
        report: dict[str, Any],
    ) -> tuple[float, float, float] | None:
        """Read cumulative GPU-hours and cost from the current report format."""

        metrics = report.get("metrics")
        if not isinstance(metrics, dict):
            return None
        trainer_gpu_hours = self._nonnegative_float(
            metrics.get("fireworks/usage/trainer_gpu_hours")
        )
        rollout_gpu_hours = self._nonnegative_float(
            metrics.get("fireworks/usage/rollout_gpu_hours")
        )
        gpu_cost_usd = self._nonnegative_float(
            metrics.get("fireworks/estimated_cost/gpu_total_usd")
        )
        values = (trainer_gpu_hours, rollout_gpu_hours, gpu_cost_usd)
        if all(value is None for value in values):
            return None
        if any(value is None for value in values):
            raise RuntimeError(
                "Fireworks checkpoint usage has incomplete GPU-hour accounting"
            )
        assert trainer_gpu_hours is not None
        assert rollout_gpu_hours is not None
        assert gpu_cost_usd is not None
        return trainer_gpu_hours, rollout_gpu_hours, gpu_cost_usd

    def record_external_samples(
        self,
        *,
        sampling_requests: int,
        prompt_tokens: int,
        prompt_cache_hit_tokens: int,
        prompt_cache_unknown_tokens: int = 0,
        sampled_tokens: int,
        elapsed_s: float,
    ) -> None:
        """Record token usage from out-of-process Harbor rollout workers."""

        if (
            min(
                sampling_requests,
                prompt_tokens,
                prompt_cache_hit_tokens,
                prompt_cache_unknown_tokens,
                sampled_tokens,
            )
            < 0
            or elapsed_s < 0
        ):
            raise ValueError("Fireworks sampling usage must be non-negative")
        if prompt_cache_hit_tokens + prompt_cache_unknown_tokens > prompt_tokens:
            raise ValueError(
                "Fireworks cached plus unknown prompt tokens cannot exceed total "
                "prompt tokens"
            )
        with self._usage_lock:
            self._sampling_requests += sampling_requests
            self._prompt_tokens += prompt_tokens
            self._prompt_cache_hit_tokens += prompt_cache_hit_tokens
            self._prompt_cache_unknown_tokens += prompt_cache_unknown_tokens
            self._sampled_tokens += sampled_tokens
            self._sampling_request_seconds += elapsed_s

    def record_forward_backward(
        self,
        *,
        training_tokens: int,
        elapsed_s: float,
        succeeded: bool,
    ) -> None:
        """Record one Fireworks trainer forward/backward request."""

        if training_tokens < 0 or elapsed_s < 0:
            raise ValueError("Fireworks training usage must be non-negative")
        with self._usage_lock:
            self._forward_backward_calls += 1
            self._forward_backward_seconds += elapsed_s
            if succeeded:
                self._training_tokens += training_tokens

    def usage_metrics(self) -> dict[str, int | float]:
        """Return cumulative numeric GPU-hour estimates for W&B.

        Fireworks bills dedicated RFT and rollout capacity by active GPU
        second. The SDK does not expose exact billing-state transitions, so the
        local estimate applies each run segment's configured topology to its
        driver-observed interval beginning immediately before provisioning.
        """

        with self._usage_lock:
            current_elapsed_s = max(0.0, time.monotonic() - self._started_monotonic)
            elapsed_s = self._restored_wall_time_seconds + current_elapsed_s
            metrics: dict[str, int | float] = {
                "fireworks/usage/wall_time_seconds": elapsed_s,
                "fireworks/usage/restored_wall_time_seconds": self._restored_wall_time_seconds,
                "fireworks/usage/current_stage_wall_time_seconds": current_elapsed_s,
                "fireworks/usage/sampling_requests_total": self._sampling_requests,
                "fireworks/usage/prompt_tokens_total": self._prompt_tokens,
                "fireworks/usage/prompt_cache_hit_tokens_total": (
                    self._prompt_cache_hit_tokens
                ),
                "fireworks/usage/prompt_cache_unknown_tokens_total": (
                    self._prompt_cache_unknown_tokens
                ),
                "fireworks/usage/sampled_tokens_total": self._sampled_tokens,
                "fireworks/usage/sampling_request_seconds_total": (
                    self._sampling_request_seconds
                ),
                "fireworks/usage/forward_backward_calls_total": (
                    self._forward_backward_calls
                ),
                "fireworks/usage/training_tokens_total": self._training_tokens,
                "fireworks/usage/forward_backward_seconds_total": (
                    self._forward_backward_seconds
                ),
            }
        trainer_gpus_per_replica = self.config.billing_trainer_gpus_per_replica
        rollout_gpus_per_replica = self.config.billing_rollout_gpus_per_replica
        gpu_price = self.config.billing_gpu_price_per_hour_usd
        if (
            trainer_gpus_per_replica is None
            or rollout_gpus_per_replica is None
            or gpu_price is None
        ):
            return metrics

        trainer_gpu_count = self.config.trainer_replica_count * trainer_gpus_per_replica
        rollout_gpu_count = self.config.replica_count * rollout_gpus_per_replica
        total_gpu_count = trainer_gpu_count + rollout_gpu_count
        current_hours = current_elapsed_s / 3600.0
        current_trainer_gpu_hours = trainer_gpu_count * current_hours
        current_rollout_gpu_hours = rollout_gpu_count * current_hours
        trainer_gpu_hours = self._restored_trainer_gpu_hours + current_trainer_gpu_hours
        rollout_gpu_hours = self._restored_rollout_gpu_hours + current_rollout_gpu_hours
        total_gpu_hours = trainer_gpu_hours + rollout_gpu_hours
        current_gpu_cost_usd = (
            current_trainer_gpu_hours + current_rollout_gpu_hours
        ) * gpu_price
        total_gpu_cost_usd = self._restored_gpu_cost_usd + current_gpu_cost_usd
        metrics.update(
            {
                "fireworks/usage/trainer_gpu_count": trainer_gpu_count,
                "fireworks/usage/rollout_gpu_count": rollout_gpu_count,
                "fireworks/usage/total_gpu_count": total_gpu_count,
                "fireworks/usage/trainer_gpu_hours": trainer_gpu_hours,
                "fireworks/usage/rollout_gpu_hours": rollout_gpu_hours,
                "fireworks/usage/total_gpu_hours": total_gpu_hours,
                "fireworks/usage/current_stage_trainer_gpu_hours": current_trainer_gpu_hours,
                "fireworks/usage/current_stage_rollout_gpu_hours": current_rollout_gpu_hours,
                "fireworks/usage/current_stage_total_gpu_hours": (
                    current_trainer_gpu_hours + current_rollout_gpu_hours
                ),
                "fireworks/usage/billing_stage_count": len(
                    self._restored_billing_stages
                )
                + 1,
                "fireworks/usage/cost_history_complete": int(
                    self._cost_history_complete
                ),
                "fireworks/cost/gpu_price_per_hour_usd": gpu_price,
                "fireworks/estimated_cost/current_stage_gpu_usd": current_gpu_cost_usd,
                "fireworks/estimated_cost/gpu_total_usd": total_gpu_cost_usd,
            }
        )
        return metrics

    def usage_summary(self) -> dict[str, int | float | str]:
        """Return final W&B summary values and the cost estimate basis."""

        summary: dict[str, int | float | str] = dict(self.usage_metrics())
        summary.update(
            {
                "fireworks/run/started_at_utc": self._started_at_utc,
                "fireworks/run/base_model": self.config.base_model or "",
                "fireworks/run/training_shape_id": (
                    self.config.training_shape_id or ""
                ),
                "fireworks/run/trainer_job_id": self.trainer_job_id or "",
                "fireworks/run/deployment_id": self.deployment_id or "",
                "fireworks/run/cost_estimate_basis": (
                    "sum of per-run billing segments, each using its recorded "
                    "trainer/rollout topology, driver-observed active time, and "
                    "GPU-hour rate; provider invoice is authoritative"
                ),
            }
        )
        if self.config.billing_gpu_type:
            summary["fireworks/run/gpu_type"] = self.config.billing_gpu_type
        return summary

    def usage_report(self) -> dict[str, Any]:
        """Return a JSON-serializable audit record for checkpoint manifests."""

        metrics = self.usage_metrics()
        current_stage: dict[str, Any] | None = None
        if "fireworks/usage/current_stage_trainer_gpu_hours" in metrics:
            current_stage = {
                "started_at_utc": self._current_stage_started_at_utc,
                "wall_time_seconds": metrics[
                    "fireworks/usage/current_stage_wall_time_seconds"
                ],
                "gpu_type": self.config.billing_gpu_type,
                "trainer_replica_count": self.config.trainer_replica_count,
                "trainer_gpus_per_replica": self.config.billing_trainer_gpus_per_replica,
                "rollout_replica_count": self.config.replica_count,
                "rollout_gpus_per_replica": self.config.billing_rollout_gpus_per_replica,
                "gpu_price_per_hour_usd": self.config.billing_gpu_price_per_hour_usd,
                "trainer_gpu_hours": metrics[
                    "fireworks/usage/current_stage_trainer_gpu_hours"
                ],
                "rollout_gpu_hours": metrics[
                    "fireworks/usage/current_stage_rollout_gpu_hours"
                ],
                "estimated_cost_usd": metrics[
                    "fireworks/estimated_cost/current_stage_gpu_usd"
                ],
                "source": "current_process",
            }
        billing_stages = [dict(stage) for stage in self._restored_billing_stages]
        if current_stage is not None:
            billing_stages.append(current_stage)
        return {
            "cumulative_across_resumes": True,
            "started_at_utc": self._started_at_utc,
            "base_model": self.config.base_model,
            "training_shape_id": self.config.training_shape_id,
            "trainer_job_id": self.trainer_job_id,
            "deployment_id": self.deployment_id,
            "gpu_type": self.config.billing_gpu_type,
            "trainer_replica_count": self.config.trainer_replica_count,
            "trainer_gpus_per_replica": (self.config.billing_trainer_gpus_per_replica),
            "rollout_replica_count": self.config.replica_count,
            "rollout_gpus_per_replica": (self.config.billing_rollout_gpus_per_replica),
            "gpu_price_per_hour_usd": (self.config.billing_gpu_price_per_hour_usd),
            "estimate_basis": (
                "sum of per-run billing segments, each using its recorded "
                "trainer/rollout topology, driver-observed active time, and "
                "GPU-hour rate; provider invoice is authoritative"
            ),
            "billing_stages": billing_stages,
            "metrics": metrics,
        }

    def _snapshot_name(self, version: int) -> str:
        # Dedicated checkpoint names are lowercase DNS labels.
        prefix = (
            re.sub(r"[^a-z0-9-]+", "-", self.config.snapshot_prefix.lower()).strip("-")
            or "skyrl"
        )
        suffix = f"-v{version:08d}-{uuid.uuid4().hex[:8]}"
        # Fireworks appends another ``-<8 hex>`` suffix. Keep our input at 54
        # characters so the provider-side name remains at most 63 characters.
        prefix = prefix[: 54 - len(suffix)].rstrip("-") or "skyrl"
        return f"{prefix}{suffix}"

    async def publish_sampler_weights(self) -> SamplerVersion:
        """Save current weights and hot-load them into the stable deployment."""

        async with self._publish_lock:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("Fireworks runtime is closed")
                version = self._next_version

            name = self._snapshot_name(version)

            def _save() -> str:
                future = self.training_client.save_weights_for_sampler(name)
                result = future.result(timeout=self.config.request_timeout_s)
                path = getattr(result, "path", None)
                if not path:
                    raise RuntimeError(
                        f"Fireworks save_weights_for_sampler({name!r}) returned no path"
                    )
                return str(path)

            snapshot_path = await asyncio.to_thread(_save)
            await asyncio.to_thread(
                self.service.hotload_sampler_snapshot, snapshot_path
            )

            # The client is created once, after the first snapshot is ready.
            with self._state_lock:
                needs_sampler = self._sampler is None
            if needs_sampler:
                sampler = await asyncio.to_thread(
                    self.service.create_sampling_client,
                    tokenizer=self.tokenizer,
                )
                with self._state_lock:
                    self._sampler = sampler

            identity = SamplerVersion(version=version, snapshot_path=snapshot_path)
            with self._state_lock:
                self._sampler_identity = identity
                self._next_version = version + 1
            return identity

    @contextmanager
    def _use_sampler(self) -> Iterator[tuple[Any, SamplerVersion]]:
        """Keep the stable sampling client open for one active request."""

        with self._state_lock:
            if self._closed:
                raise RuntimeError("Fireworks runtime is closed")
            if self._sampler is None or self._sampler_identity is None:
                raise RuntimeError(
                    "Fireworks sampler weights have not been published yet"
                )
            sampler = self._sampler
            identity = self._sampler_identity
            self._active_samples += 1
        try:
            yield sampler, identity
        finally:
            with self._state_lock:
                self._active_samples -= 1
                self._state_lock.notify_all()

    async def sample_async(
        self,
        *,
        prompt: Any,
        sampling_params: Any,
    ) -> tuple[Any, SamplerVersion]:
        """Sample from the deployment without consuming an executor thread.

        The active-call scope covers the whole stream, including a provider-side
        pause/resume during an asynchronous hot-load.
        """

        with self._use_sampler() as (sampler, identity):
            native_sample = getattr(sampler, "sample_async", None)
            if native_sample is None:
                raise RuntimeError(
                    "The dedicated Fireworks sampler must expose sample_async()"
                )
            result = await asyncio.wait_for(
                native_sample(
                    prompt=prompt,
                    num_samples=1,
                    sampling_params=sampling_params,
                ),
                timeout=self.config.sampling_timeout_s,
            )
            return result, identity

    async def close(self) -> None:
        """Drain active calls, then close the sampler and service. Idempotent."""

        # Publication and teardown must not mutate the deployment concurrently.
        async with self._publish_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True

            def _wait_for_samples() -> tuple[Any | None, int]:
                deadline = time.monotonic() + self.config.sampling_timeout_s
                with self._state_lock:
                    while self._active_samples > 0:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._state_lock.wait(timeout=remaining)
                    sampler = self._sampler
                    active_samples = self._active_samples
                    self._sampler = None
                    self._sampler_identity = None
                    return sampler, active_samples

            sampler, active_calls = await asyncio.to_thread(_wait_for_samples)
            if active_calls:
                warnings.warn(
                    f"Closing Fireworks sampler with {active_calls} call(s) still "
                    f"active after sampling_timeout_s={self.config.sampling_timeout_s}",
                    RuntimeWarning,
                )
            if sampler is not None:
                await asyncio.to_thread(_close_quietly, sampler)
            await asyncio.to_thread(_close_quietly, self.service)
