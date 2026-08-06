"""Shared lifecycle for hosted Tinker training and versioned sampling."""

from __future__ import annotations

import asyncio
import inspect
import os
import threading
import time
import uuid
import warnings
from collections.abc import Awaitable, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from loguru import logger

from skyrl.train.config import TinkerTrainingConfig

_SERVICE_BOOTSTRAP_BACKOFF_CAP_S = 30.0


def configure_tinker_pyqwest_system_certs() -> None:
    """Keep Tinker TLS verification on while adding operating-system CA roots."""

    import pyqwest
    import tinker._base_client as tinker_base_client
    from pyqwest.httpx import AsyncPyqwestTransport

    def _system_ca_pyqwest_transport():
        transport_kwargs: dict[str, bool] = {}
        # pyqwest 0.7 stopped trusting system roots by default and added this
        # switch. Older versions (including the Python 3.12 lockfile version)
        # trust roots by default and reject the new keyword.
        if "tls_include_system_certs" in inspect.signature(pyqwest.HTTPTransport).parameters:
            transport_kwargs["tls_include_system_certs"] = True
        return AsyncPyqwestTransport(transport=pyqwest.HTTPTransport(**transport_kwargs))

    tinker_base_client._default_pyqwest_transport = _system_ca_pyqwest_transport


@dataclass(frozen=True)
class TinkerSamplerVersion:
    """Identity of one immutable hosted sampling policy."""

    version: int
    sampling_session_id: str | None = None
    model_path: str | None = None


def _close_service_quietly(service: Any) -> None:
    """Close the SDK's shared holder without depending on private sampler APIs."""

    holder = getattr(service, "holder", None)
    close = getattr(holder, "close", None)
    if close is None:
        close = getattr(service, "close", None)
    if close is None:
        return
    try:
        close()
    except Exception:  # noqa: BLE001  # pragma: no cover - provider teardown
        warnings.warn(
            "Failed to close Tinker service "
            f"{type(service).__name__}; exception details omitted because "
            "provider errors may contain credential-bearing headers.",
            RuntimeWarning,
        )


def _create_service_client_with_bootstrap_retry(
    *,
    tinker_module: Any,
    service_kwargs: dict[str, Any],
    max_attempts: int,
    initial_backoff_s: float,
) -> Any:
    """Retry only the SDK's initial ServiceClient connection bootstrap."""

    for attempt in range(1, max_attempts + 1):
        try:
            return tinker_module.ServiceClient(**service_kwargs)
        except tinker_module.APIConnectionError:
            if attempt >= max_attempts:
                raise RuntimeError(
                    "Tinker ServiceClient bootstrap failed after "
                    f"{max_attempts} connection attempts. Provider exception "
                    "details are omitted because transport errors may contain "
                    "credential-bearing headers."
                ) from None
            backoff_s = min(
                initial_backoff_s * (2 ** (attempt - 1)),
                _SERVICE_BOOTSTRAP_BACKOFF_CAP_S,
            )
            logger.warning(
                f"Tinker ServiceClient bootstrap connection failure on attempt "
                f"{attempt}/{max_attempts}; retrying in {backoff_s:.1f}s before "
                "any training client is created. Exception details are omitted "
                "because transport errors may contain credential-bearing headers."
            )
            time.sleep(backoff_s)

    raise AssertionError("Tinker ServiceClient bootstrap retry loop was empty")


class TinkerRuntime:
    """Own one Tinker service, training client, and published sampler.

    Publishing creates a new immutable ``SamplingClient`` and swaps it under a
    lock. A request captures its client and version before starting, so an
    in-flight request stays on the old weights while later requests use the new
    client. Tinker does not need or expose vLLM-style pause/resume controls.
    """

    _RESTORABLE_USAGE_METRICS = {
        "tinker/usage/sampling_requests_total": "_sampling_requests",
        "tinker/usage/sampling_failures_total": "_sampling_failures",
        "tinker/usage/prompt_tokens_total": "_prompt_tokens",
        "tinker/usage/prompt_cache_hit_tokens_total": "_prompt_cache_hit_tokens",
        "tinker/usage/prompt_cache_unknown_tokens_total": "_prompt_cache_unknown_tokens",
        "tinker/usage/sampled_tokens_total": "_sampled_tokens",
        "tinker/usage/sampling_request_seconds_total": "_sampling_request_seconds",
        "tinker/usage/forward_backward_calls_total": "_forward_backward_calls",
        "tinker/usage/forward_backward_failures_total": "_forward_backward_failures",
        "tinker/usage/training_tokens_total": "_training_tokens",
        "tinker/usage/forward_backward_seconds_total": "_forward_backward_seconds",
        "tinker/usage/optimizer_steps_total": "_optimizer_steps",
        "tinker/usage/optimizer_failures_total": "_optimizer_failures",
        "tinker/usage/optimizer_seconds_total": "_optimizer_seconds",
        "tinker/usage/sampler_publications_total": "_sampler_publications",
        "tinker/usage/sampler_publication_failures_total": "_sampler_publication_failures",
        "tinker/usage/sampler_publication_seconds_total": "_sampler_publication_seconds",
        "tinker/usage/checkpoint_seconds_total": "_checkpoint_seconds",
    }

    def __init__(
        self,
        *,
        service: Any,
        training_client: Any,
        tokenizer: Any,
        config: TinkerTrainingConfig,
    ) -> None:
        self.service = service
        self.training_client = training_client
        self.tokenizer = tokenizer
        self.config = config
        self._state_lock = threading.Condition()
        self._publish_lock = asyncio.Lock()
        self._sampler: Any | None = None
        self._sampler_identity: TinkerSamplerVersion | None = None
        self._sampler_publish_listeners: list[Callable[[TinkerSamplerVersion], Awaitable[None] | None]] = []
        self._next_version = 0
        self._active_samples = 0
        self._closed = False
        self._usage_lock = threading.Lock()
        self._started_at_utc = datetime.now(UTC).isoformat()
        self._started_monotonic = time.monotonic()
        self._restored_wall_time_seconds = 0.0
        self._usage_restored = False
        self._training_run_ids: list[str] = []
        self._sampling_requests = 0
        self._sampling_failures = 0
        self._prompt_tokens = 0
        self._prompt_cache_hit_tokens = 0
        self._prompt_cache_unknown_tokens = 0
        self._sampled_tokens = 0
        self._sampling_request_seconds = 0.0
        self._forward_backward_calls = 0
        self._forward_backward_failures = 0
        self._training_tokens = 0
        self._forward_backward_seconds = 0.0
        self._optimizer_steps = 0
        self._optimizer_failures = 0
        self._optimizer_seconds = 0.0
        self._sampler_publications = 0
        self._sampler_publication_failures = 0
        self._sampler_publication_seconds = 0.0
        self._checkpoint_seconds = 0.0
        self._checkpoints: list[dict[str, Any]] = []
        self._last_sampling_session_id: str | None = None
        self._validate_cost_watchdog_config()

    def _validate_cost_watchdog_config(self) -> None:
        limit = self.config.max_estimated_cost_usd
        if limit is None:
            return
        if limit <= 0:
            raise ValueError("Tinker max_estimated_cost_usd must be positive")
        missing_prices = [
            name
            for name in (
                "prefill_price_per_million_tokens",
                "sample_price_per_million_tokens",
                "train_price_per_million_tokens",
            )
            if getattr(self.config, name) is None
        ]
        if missing_prices:
            raise ValueError("Tinker max_estimated_cost_usd requires token prices: " + ", ".join(missing_prices))

    def _estimated_token_cost_usd_locked(self) -> float | None:
        prefill_price = self.config.prefill_price_per_million_tokens
        sample_price = self.config.sample_price_per_million_tokens
        train_price = self.config.train_price_per_million_tokens
        if prefill_price is None or sample_price is None or train_price is None:
            return None
        cached_prefill_price = self.config.cached_prefill_price_per_million_tokens
        if cached_prefill_price is None:
            cached_prefill_price = prefill_price * 0.2
        prompt_cache_known_tokens = max(0, self._prompt_tokens - self._prompt_cache_unknown_tokens)
        prompt_uncached_tokens = max(
            0,
            prompt_cache_known_tokens - self._prompt_cache_hit_tokens,
        )
        # Prompt tokens with unknown cache status are conservatively priced as
        # uncached.
        prefill_cost = (
            self._prompt_cache_hit_tokens * cached_prefill_price
            + (prompt_uncached_tokens + self._prompt_cache_unknown_tokens) * prefill_price
        ) / 1_000_000
        sampling_cost = self._sampled_tokens * sample_price / 1_000_000
        training_cost = self._training_tokens * train_price / 1_000_000
        return prefill_cost + sampling_cost + training_cost

    def _enforce_cost_watchdog_locked(self) -> None:
        limit = self.config.max_estimated_cost_usd
        if limit is None:
            return
        estimated_cost = self._estimated_token_cost_usd_locked()
        if estimated_cost is None:
            raise RuntimeError("Tinker estimated-cost watchdog is missing configured token prices")
        if estimated_cost > limit:
            raise RuntimeError(
                "Tinker estimated token cost "
                f"${estimated_cost:.2f} exceeded the configured "
                f"${limit:.2f} watchdog limit. Recorded usage is preserved; "
                "in-flight provider calls may add further cost."
            )

    def _current_price_config(self) -> dict[str, float | None]:
        prefill_price = self.config.prefill_price_per_million_tokens
        cached_prefill_price = self.config.cached_prefill_price_per_million_tokens
        if cached_prefill_price is None and prefill_price is not None:
            cached_prefill_price = prefill_price * 0.2
        return {
            "prefill_uncached": prefill_price,
            "prefill_cached": cached_prefill_price,
            "sample": self.config.sample_price_per_million_tokens,
            "train": self.config.train_price_per_million_tokens,
        }

    def _validate_restored_price_config(self, report: dict[str, Any]) -> None:
        saved_prices = report.get("price_per_million_tokens")
        if not isinstance(saved_prices, dict):
            raise RuntimeError(
                "Tinker checkpoint usage is missing price_per_million_tokens; "
                "only the current accounting format is supported"
            )
        current_prices = self._current_price_config()
        mismatches = [
            name
            for name, current_value in current_prices.items()
            if saved_prices.get(name) != current_value
        ]
        if mismatches:
            raise RuntimeError(
                "Tinker usage restore pricing does not match the checkpoint for: "
                + ", ".join(mismatches)
                + ". Resume with the recorded prices so historical cost is not "
                "silently repriced."
            )

    def restore_usage_reports(self, reports: list[dict[str, Any]]) -> None:
        """Restore cumulative usage before a resumed process starts provider work."""

        if (
            len(reports) != 1
            or reports[0].get("cumulative_across_resumes") is not True
        ):
            raise RuntimeError(
                "Tinker resume requires one cumulative current-format usage report"
            )

        with self._usage_lock:
            if self._usage_restored:
                raise RuntimeError("Tinker usage has already been restored")
            if any(
                (
                    self._sampling_requests,
                    self._forward_backward_calls,
                    self._optimizer_steps,
                    self._sampler_publications,
                    self._checkpoints,
                )
            ):
                raise RuntimeError("Tinker usage must be restored before recording provider work")

            for report in reports:
                self._validate_restored_price_config(report)

            restored_checkpoints: list[dict[str, Any]] = []
            restored_run_ids: list[str] = []
            started_at_values: list[str] = []
            for report in reports:
                metrics = report.get("metrics")
                if not isinstance(metrics, dict):
                    raise RuntimeError("Tinker checkpoint usage metrics must be an object")
                required_cache_metrics = {
                    "tinker/usage/prompt_cache_hit_tokens_total",
                    "tinker/usage/prompt_cache_unknown_tokens_total",
                }
                missing_cache_metrics = sorted(required_cache_metrics - metrics.keys())
                if missing_cache_metrics:
                    raise RuntimeError(
                        "Tinker checkpoint usage is missing cache accounting metrics: "
                        + ", ".join(missing_cache_metrics)
                    )
                self._restored_wall_time_seconds += float(metrics.get("tinker/usage/wall_time_seconds", 0.0))
                for (
                    metric_name,
                    attribute_name,
                ) in self._RESTORABLE_USAGE_METRICS.items():
                    value = metrics.get(metric_name, 0)
                    current = getattr(self, attribute_name)
                    setattr(self, attribute_name, current + value)

                checkpoints = report.get("checkpoints")
                if isinstance(checkpoints, list):
                    restored_checkpoints.extend(dict(record) for record in checkpoints if isinstance(record, dict))

                report_run_ids = report.get("training_run_ids")
                if not isinstance(report_run_ids, list):
                    report_run_ids = [report.get("training_run_id")]
                for run_id in report_run_ids:
                    if run_id and str(run_id) not in restored_run_ids:
                        restored_run_ids.append(str(run_id))

                started_at = report.get("started_at_utc")
                if started_at:
                    started_at_values.append(str(started_at))

            current_run_id = self.training_run_id
            if current_run_id and current_run_id not in restored_run_ids:
                restored_run_ids.append(current_run_id)
            self._training_run_ids = restored_run_ids
            self._checkpoints = restored_checkpoints
            if started_at_values:
                self._started_at_utc = min(started_at_values)
            self._usage_restored = True
            self._enforce_cost_watchdog_locked()

    @classmethod
    def connect(
        cls,
        *,
        config: TinkerTrainingConfig,
        tokenizer: Any,
        lora_rank: int,
        run_name: str | None = None,
    ) -> TinkerRuntime:
        """Create a hosted LoRA training session.

        This is the provider preflight and may allocate billable resources.
        Callers should invoke it only after validating the complete SkyRL
        configuration.
        """

        if not os.environ.get("TINKER_API_KEY"):
            raise RuntimeError("TINKER_API_KEY is required for hosted Tinker training")
        if not config.base_model:
            raise ValueError("trainer.tinker.base_model is required")
        if lora_rank <= 0:
            raise ValueError("Hosted Tinker training currently requires LoRA rank > 0")
        if config.service_bootstrap_max_attempts <= 0:
            raise ValueError("Tinker service_bootstrap_max_attempts must be positive")
        if config.service_bootstrap_retry_backoff_s < 0:
            raise ValueError("Tinker service_bootstrap_retry_backoff_s must be non-negative")
        if not any((config.train_mlp, config.train_attn, config.train_unembed)):
            raise ValueError("At least one of trainer.tinker.train_mlp, train_attn, or " "train_unembed must be true")

        try:
            import tinker
        except ImportError as exc:
            raise ImportError(
                "The hosted Tinker backend requires the tinker package; "
                "install SkyRL with --extra tinker --extra skyrl-train"
            ) from exc

        # Tinker SDK 0.23 enables its pyqwest transport through a server-side
        # feature flag. Its default transport does not include the operating
        # system CA store, which breaks TLS behind otherwise trusted enterprise
        # certificate chains (for example, macOS curl succeeds while pyqwest
        # raises UnknownIssuer). Keep verification enabled and add system roots.
        configure_tinker_pyqwest_system_certs()

        service_kwargs: dict[str, Any] = {
            "project_id": config.project_id,
            "user_metadata": {
                "integration": "skyrl",
                **({"run_name": run_name} if run_name else {}),
            },
        }
        if config.base_url:
            service_kwargs["base_url"] = config.base_url

        service = _create_service_client_with_bootstrap_retry(
            tinker_module=tinker,
            service_kwargs=service_kwargs,
            max_attempts=config.service_bootstrap_max_attempts,
            initial_backoff_s=config.service_bootstrap_retry_backoff_s,
        )
        try:
            training_client = service.create_lora_training_client(
                base_model=config.base_model,
                rank=lora_rank,
                seed=config.seed,
                train_mlp=config.train_mlp,
                train_attn=config.train_attn,
                train_unembed=config.train_unembed,
                user_metadata={"integration": "skyrl"},
            )
        except BaseException:
            _close_service_quietly(service)
            raise

        return cls(
            service=service,
            training_client=training_client,
            tokenizer=tokenizer,
            config=config,
        )

    @property
    def model_name(self) -> str:
        assert self.config.base_model is not None
        return self.config.base_model

    @property
    def endpoint_url(self) -> str:
        return self.config.base_url or "tinker://hosted"

    @property
    def weight_version(self) -> int:
        with self._state_lock:
            return -1 if self._sampler_identity is None else self._sampler_identity.version

    @property
    def training_run_id(self) -> str:
        """Return Tinker's public model/training-run identifier."""

        return str(getattr(self.training_client, "model_id", "") or "")

    def add_sampler_publish_listener(
        self,
        listener: Callable[[TinkerSamplerVersion], Awaitable[None] | None],
    ) -> None:
        """Register a correctness-critical observer of sampler publication.

        Listeners are awaited before :meth:`publish_sampler_weights` returns.
        This lets out-of-process agent workers observe the new immutable model
        path at the same publication boundary as the normal inference client.
        """

        with self._state_lock:
            if self._closed:
                raise RuntimeError("Tinker runtime is closed")
            self._sampler_publish_listeners.append(listener)

    def _record_sample(
        self,
        *,
        prompt_tokens: int,
        prompt_cache_hit_tokens: int,
        prompt_cache_unknown_tokens: int = 0,
        sampled_tokens: int,
        elapsed_s: float,
        succeeded: bool,
    ) -> None:
        if (
            min(
                prompt_tokens,
                prompt_cache_hit_tokens,
                prompt_cache_unknown_tokens,
                sampled_tokens,
            )
            < 0
            or elapsed_s < 0
        ):
            raise ValueError("Tinker sampling usage must be non-negative")
        if prompt_cache_hit_tokens + prompt_cache_unknown_tokens > prompt_tokens:
            raise ValueError("Tinker cached plus unknown prompt tokens cannot exceed total " "prompt tokens")
        with self._usage_lock:
            self._sampling_requests += 1
            self._sampling_request_seconds += elapsed_s
            if succeeded:
                self._prompt_tokens += prompt_tokens
                self._prompt_cache_hit_tokens += prompt_cache_hit_tokens
                self._prompt_cache_unknown_tokens += prompt_cache_unknown_tokens
                self._sampled_tokens += sampled_tokens
                self._enforce_cost_watchdog_locked()
            else:
                self._sampling_failures += 1

    def record_forward_backward(
        self,
        *,
        training_tokens: int,
        elapsed_s: float,
        succeeded: bool,
    ) -> None:
        """Record one logical Tinker forward/backward request."""

        with self._usage_lock:
            self._forward_backward_calls += 1
            self._forward_backward_seconds += elapsed_s
            if succeeded:
                self._training_tokens += training_tokens
                self._enforce_cost_watchdog_locked()
            else:
                self._forward_backward_failures += 1

    def record_optimizer_step(self, *, elapsed_s: float, succeeded: bool) -> None:
        with self._usage_lock:
            self._optimizer_steps += 1
            self._optimizer_seconds += elapsed_s
            if not succeeded:
                self._optimizer_failures += 1

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
        """Record successful samples made by out-of-process agent workers.

        Apex/Harbor workers create a Tinker client from a published ``model_path``
        and therefore bypass :meth:`sample_async`. They return exact token
        counts to the driver, which uses this method to keep the normal Tinker
        usage and cost metrics complete.
        """

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
            raise ValueError("External Tinker sampling usage must be non-negative")
        if prompt_cache_hit_tokens + prompt_cache_unknown_tokens > prompt_tokens:
            raise ValueError("External Tinker cached plus unknown prompt tokens cannot exceed " "total prompt tokens")
        with self._usage_lock:
            self._sampling_requests += sampling_requests
            self._prompt_tokens += prompt_tokens
            self._prompt_cache_hit_tokens += prompt_cache_hit_tokens
            self._prompt_cache_unknown_tokens += prompt_cache_unknown_tokens
            self._sampled_tokens += sampled_tokens
            self._sampling_request_seconds += elapsed_s
            self._enforce_cost_watchdog_locked()

    def record_checkpoint(
        self,
        *,
        global_step: int | None,
        checkpoint_name: str,
        provider_path: str,
        elapsed_s: float,
    ) -> None:
        with self._usage_lock:
            self._checkpoint_seconds += elapsed_s
            self._checkpoints.append(
                {
                    "global_step": global_step,
                    "checkpoint_name": checkpoint_name,
                    "provider_path": provider_path,
                    "created_at_utc": datetime.now(UTC).isoformat(),
                }
            )

    def usage_metrics(self) -> dict[str, int | float]:
        """Return cumulative numeric metrics suitable for W&B time series."""

        with self._usage_lock:
            prompt_tokens = self._prompt_tokens
            prompt_cache_hit_tokens = self._prompt_cache_hit_tokens
            prompt_cache_unknown_tokens = self._prompt_cache_unknown_tokens
            sampled_tokens = self._sampled_tokens
            training_tokens = self._training_tokens
            prompt_cache_known_tokens = max(0, prompt_tokens - prompt_cache_unknown_tokens)
            prompt_uncached_tokens = max(
                0,
                prompt_cache_known_tokens - prompt_cache_hit_tokens,
            )
            metrics: dict[str, int | float] = {
                "tinker/usage/wall_time_seconds": self._restored_wall_time_seconds
                + time.monotonic()
                - self._started_monotonic,
                "tinker/usage/sampling_requests_total": self._sampling_requests,
                "tinker/usage/sampling_failures_total": self._sampling_failures,
                "tinker/usage/prompt_tokens_total": prompt_tokens,
                "tinker/usage/prompt_cache_hit_tokens_total": prompt_cache_hit_tokens,
                "tinker/usage/prompt_uncached_tokens_total": prompt_uncached_tokens,
                "tinker/usage/prompt_cache_unknown_tokens_total": prompt_cache_unknown_tokens,
                "tinker/usage/sampled_tokens_total": sampled_tokens,
                "tinker/usage/sampling_request_seconds_total": self._sampling_request_seconds,
                "tinker/usage/forward_backward_calls_total": self._forward_backward_calls,
                "tinker/usage/forward_backward_failures_total": self._forward_backward_failures,
                "tinker/usage/training_tokens_total": training_tokens,
                "tinker/usage/forward_backward_seconds_total": self._forward_backward_seconds,
                "tinker/usage/optimizer_steps_total": self._optimizer_steps,
                "tinker/usage/optimizer_failures_total": self._optimizer_failures,
                "tinker/usage/optimizer_seconds_total": self._optimizer_seconds,
                "tinker/usage/sampler_publications_total": self._sampler_publications,
                "tinker/usage/sampler_publication_failures_total": self._sampler_publication_failures,
                "tinker/usage/sampler_publication_seconds_total": self._sampler_publication_seconds,
                "tinker/usage/checkpoints_total": len(self._checkpoints),
                "tinker/usage/checkpoint_seconds_total": self._checkpoint_seconds,
            }
            if prompt_cache_known_tokens:
                metrics["tinker/usage/prompt_cache_hit_rate"] = prompt_cache_hit_tokens / prompt_cache_known_tokens

        prefill_price = self.config.prefill_price_per_million_tokens
        cached_prefill_price = self.config.cached_prefill_price_per_million_tokens
        if cached_prefill_price is None and prefill_price is not None:
            cached_prefill_price = prefill_price * 0.2
        sample_price = self.config.sample_price_per_million_tokens
        train_price = self.config.train_price_per_million_tokens
        if prefill_price is not None:
            cached_prefill_cost = (
                0.0 if cached_prefill_price is None else prompt_cache_hit_tokens * cached_prefill_price / 1_000_000
            )
            uncached_prefill_cost = prompt_uncached_tokens * prefill_price / 1_000_000
            unknown_prefill_upper_bound = prompt_cache_unknown_tokens * prefill_price / 1_000_000
            metrics["tinker/estimated_cost/prefill_cached_usd"] = cached_prefill_cost
            metrics["tinker/estimated_cost/prefill_uncached_usd"] = uncached_prefill_cost
            metrics["tinker/estimated_cost/prefill_unknown_upper_bound_usd"] = unknown_prefill_upper_bound
            metrics["tinker/estimated_cost/prefill_usd"] = (
                cached_prefill_cost + uncached_prefill_cost + unknown_prefill_upper_bound
            )
        if sample_price is not None:
            metrics["tinker/estimated_cost/sampling_usd"] = sampled_tokens * sample_price / 1_000_000
        if train_price is not None:
            metrics["tinker/estimated_cost/training_usd"] = training_tokens * train_price / 1_000_000
        billable_costs = [
            metrics[key]
            for key in (
                "tinker/estimated_cost/prefill_usd",
                "tinker/estimated_cost/sampling_usd",
                "tinker/estimated_cost/training_usd",
            )
            if key in metrics
        ]
        if billable_costs:
            metrics["tinker/estimated_cost/token_total_usd"] = float(sum(billable_costs))
        cost_limit = self.config.max_estimated_cost_usd
        token_total = metrics.get("tinker/estimated_cost/token_total_usd")
        if cost_limit is not None and token_total is not None:
            estimated_cost = float(token_total)
            metrics["tinker/estimated_cost/limit_usd"] = cost_limit
            metrics["tinker/estimated_cost/remaining_usd"] = max(0.0, cost_limit - estimated_cost)
            metrics["tinker/estimated_cost/over_limit_usd"] = max(0.0, estimated_cost - cost_limit)
        return metrics

    def usage_summary(self) -> dict[str, int | float | str]:
        """Return final W&B summary fields, including provider identifiers."""

        summary: dict[str, int | float | str] = dict(self.usage_metrics())
        summary.update(
            {
                "tinker/run/started_at_utc": self._started_at_utc,
                "tinker/run/training_run_id": self.training_run_id,
                "tinker/run/base_model": self.model_name,
                "tinker/run/cost_estimate_basis": (
                    "successful provider calls; provider-reported prompt cache hits; " "configured token prices"
                ),
            }
        )
        with self._usage_lock:
            training_run_ids = list(self._training_run_ids)
            current_run_id = self.training_run_id
            if current_run_id and current_run_id not in training_run_ids:
                training_run_ids.append(current_run_id)
            if training_run_ids:
                summary["tinker/run/training_run_ids"] = ",".join(training_run_ids)
            if self._last_sampling_session_id:
                summary["tinker/run/last_sampling_session_id"] = self._last_sampling_session_id
            if self._checkpoints:
                last_checkpoint = self._checkpoints[-1]
                summary["tinker/checkpoint/last_global_step"] = (
                    -1 if last_checkpoint["global_step"] is None else int(last_checkpoint["global_step"])
                )
                summary["tinker/checkpoint/last_name"] = str(last_checkpoint["checkpoint_name"])
                summary["tinker/checkpoint/last_provider_path"] = str(last_checkpoint["provider_path"])
        return summary

    def usage_report(self) -> dict[str, Any]:
        """Return a JSON-serializable audit record for checkpoint manifests."""

        with self._usage_lock:
            checkpoints = [dict(record) for record in self._checkpoints]
            training_run_ids = list(self._training_run_ids)
            current_run_id = self.training_run_id
            if current_run_id and current_run_id not in training_run_ids:
                training_run_ids.append(current_run_id)
        return {
            "cumulative_across_resumes": True,
            "started_at_utc": self._started_at_utc,
            "training_run_id": self.training_run_id,
            "training_run_ids": training_run_ids,
            "base_model": self.model_name,
            "price_per_million_tokens": self._current_price_config(),
            "estimate_basis": (
                "successful provider calls; provider-reported prompt cache hits; " "configured token prices"
            ),
            "metrics": self.usage_metrics(),
            "checkpoints": checkpoints,
        }

    async def publish_sampler_weights(self) -> TinkerSamplerVersion:
        """Create and atomically publish a sampler for the current weights."""

        async with self._publish_lock:
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("Tinker runtime is closed")
                version = self._next_version

            started = time.monotonic()
            try:
                model_path = None
                if self.config.publish_sampler_model_path:
                    checkpoint_name = f"skyrl-sampler-v{version:08d}-{uuid.uuid4().hex[:8]}"
                    future = await self.training_client.save_weights_for_sampler_async(
                        checkpoint_name,
                        ttl_seconds=self.config.sampler_checkpoint_ttl_seconds,
                    )
                    response = await asyncio.wait_for(
                        future.result_async(),
                        timeout=self.config.request_timeout_s,
                    )
                    model_path = str(getattr(response, "path", "") or "")
                    if not model_path:
                        raise RuntimeError(
                            f"Tinker save_weights_for_sampler({checkpoint_name!r}) returned no model path"
                        )
                    new_sampler = await asyncio.wait_for(
                        self.service.create_sampling_client_async(model_path=model_path),
                        timeout=self.config.request_timeout_s,
                    )
                else:
                    new_sampler = await asyncio.wait_for(
                        self.training_client.save_weights_and_get_sampling_client_async(),
                        timeout=self.config.request_timeout_s,
                    )
            except BaseException:
                with self._usage_lock:
                    self._sampler_publications += 1
                    self._sampler_publication_failures += 1
                    self._sampler_publication_seconds += time.monotonic() - started
                raise
            session_id = getattr(new_sampler, "_sampling_session_id", None)
            identity = TinkerSamplerVersion(
                version=version,
                sampling_session_id=str(session_id) if session_id else None,
                model_path=model_path,
            )
            with self._usage_lock:
                self._sampler_publications += 1
                self._sampler_publication_seconds += time.monotonic() - started
                self._last_sampling_session_id = identity.sampling_session_id

            # The new client is fully constructed before this atomic swap. Any
            # request that already captured the old client keeps its local ref.
            with self._state_lock:
                if self._closed:
                    raise RuntimeError("Tinker runtime closed during sampler publication")
                self._sampler = new_sampler
                self._sampler_identity = identity
                self._next_version = version + 1
                listeners = tuple(self._sampler_publish_listeners)

            # A listener failure means agent workers may not see the sampler
            # version that the trainer considers published. Fail the weight
            # publication instead of silently training with inconsistent
            # version metadata.
            for listener in listeners:
                result = listener(identity)
                if inspect.isawaitable(result):
                    await result
            return identity

    def current_sampler_identity(self) -> TinkerSamplerVersion:
        """Return the immutable sampler descriptor currently used for admission."""

        with self._state_lock:
            if self._closed:
                raise RuntimeError("Tinker runtime is closed")
            if self._sampler_identity is None:
                raise RuntimeError("Tinker sampler weights have not been published yet")
            return self._sampler_identity

    def current_sampler(self) -> tuple[Any, TinkerSamplerVersion]:
        """Capture the current picklable sampler and its identity atomically.

        Tinker's public ``SamplingClient`` is safe to share with worker
        processes.  Sharing it keeps every worker on this runtime's one SDK
        service session instead of constructing a ``ServiceClient`` per
        request or trajectory.
        """

        with self._state_lock:
            if self._closed:
                raise RuntimeError("Tinker runtime is closed")
            if self._sampler is None or self._sampler_identity is None:
                raise RuntimeError("Tinker sampler weights have not been published yet")
            return self._sampler, self._sampler_identity

    @contextmanager
    def _use_sampler(self) -> Iterator[tuple[Any, TinkerSamplerVersion]]:
        """Capture one immutable sampling client for an entire request."""

        with self._state_lock:
            if self._closed:
                raise RuntimeError("Tinker runtime is closed")
            if self._sampler is None or self._sampler_identity is None:
                raise RuntimeError("Tinker sampler weights have not been published yet")
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
        prompt_token_count: int = 0,
    ) -> tuple[Any, TinkerSamplerVersion]:
        """Sample with the exact client version captured at request admission."""

        with self._use_sampler() as (sampler, identity):
            started = time.monotonic()
            try:
                result = await asyncio.wait_for(
                    sampler.sample_async(
                        prompt=prompt,
                        num_samples=1,
                        sampling_params=sampling_params,
                    ),
                    timeout=self.config.sampling_timeout_s,
                )
            except BaseException:
                self._record_sample(
                    prompt_tokens=0,
                    prompt_cache_hit_tokens=0,
                    sampled_tokens=0,
                    elapsed_s=time.monotonic() - started,
                    succeeded=False,
                )
                raise
            sequences = getattr(result, "sequences", None)
            sampled_token_count = 0
            for sequence in () if sequences is None else sequences:
                sequence_tokens = getattr(sequence, "tokens", None)
                if sequence_tokens is not None:
                    sampled_token_count += len(sequence_tokens)
            self._record_sample(
                prompt_tokens=prompt_token_count,
                prompt_cache_hit_tokens=int(getattr(result, "prompt_cache_hit_tokens", 0) or 0),
                sampled_tokens=sampled_token_count,
                elapsed_s=time.monotonic() - started,
                succeeded=True,
            )
            return result, identity

    async def close(self) -> None:
        """Drain active samples and close the SDK session. Idempotent."""

        async with self._publish_lock:
            with self._state_lock:
                if self._closed:
                    return
                self._closed = True

            def _wait_for_samples() -> int:
                deadline = time.monotonic() + self.config.close_timeout_s
                with self._state_lock:
                    while self._active_samples > 0:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        self._state_lock.wait(timeout=remaining)
                    active = self._active_samples
                    self._sampler = None
                    self._sampler_identity = None
                    return active

            active_samples = await asyncio.to_thread(_wait_for_samples)
            if active_samples:
                warnings.warn(
                    f"Closing Tinker with {active_samples} sample call(s) still "
                    f"active after close_timeout_s={self.config.close_timeout_s}",
                    RuntimeWarning,
                )
            await asyncio.to_thread(_close_service_quietly, self.service)
