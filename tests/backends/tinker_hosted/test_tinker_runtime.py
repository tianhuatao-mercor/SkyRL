import asyncio

import httpx
import pytest

import skyrl.backends.tinker.runtime as tinker_runtime
from skyrl.backends.tinker.runtime import TinkerRuntime
from skyrl.train.config import TinkerTrainingConfig


class _Holder:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class _Service:
    def __init__(self):
        self.holder = _Holder()
        self.created_model_paths = []

    async def create_sampling_client_async(self, *, model_path):
        self.created_model_paths.append(model_path)
        return _Sampler(f"path-{model_path.rsplit('/', 1)[-1]}")


class _Sampler:
    def __init__(self, name: str):
        self.name = name
        self._sampling_session_id = f"session-{name}"
        self.started = asyncio.Event()
        self.finish = asyncio.Event()

    async def sample_async(self, *, prompt, num_samples, sampling_params):
        assert num_samples == 1
        self.started.set()
        await self.finish.wait()
        return (self.name, prompt, sampling_params)


class _TrainingClient:
    def __init__(self):
        self.samplers = []
        self.fail_next = False

    async def save_weights_and_get_sampling_client_async(self):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("publication failed")
        sampler = _Sampler(f"v{len(self.samplers)}")
        self.samplers.append(sampler)
        return sampler

    async def save_weights_for_sampler_async(self, name, *, ttl_seconds):
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("publication failed")
        path = f"tinker://training/sampler_weights/{name}"

        class _Response:
            pass

        response = _Response()
        response.path = path

        class _Future:
            async def result_async(self):
                return response

        return _Future()


def test_close_warning_omits_provider_exception_details() -> None:
    secret_marker = "must-not-appear-in-close-warning"

    class _FailingHolder:
        def close(self):
            raise RuntimeError(secret_marker)

    service = type("_FailingService", (), {"holder": _FailingHolder()})()

    with pytest.warns(RuntimeWarning) as warning_records:
        tinker_runtime._close_service_quietly(service)

    assert secret_marker not in str(warning_records[0].message)


def _runtime(
    *,
    service=None,
    training_client=None,
    config=None,
) -> TinkerRuntime:
    return TinkerRuntime(
        service=service or _Service(),
        training_client=training_client or _TrainingClient(),
        tokenizer="tokenizer",
        config=config or TinkerTrainingConfig(),
    )


def test_connect_creates_hosted_lora_session(monkeypatch) -> None:
    import pyqwest
    import tinker
    import tinker._base_client as tinker_base_client

    captured = {}
    captured_transport = {}
    service = _Service()
    training_client = _TrainingClient()

    def _service_factory(**kwargs):
        captured["service"] = kwargs
        return service

    def _create_lora(**kwargs):
        captured["training"] = kwargs
        return training_client

    service.create_lora_training_client = _create_lora
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    monkeypatch.setattr(tinker, "ServiceClient", _service_factory)

    def _http_transport(*, tls_include_system_certs=False):
        captured_transport["tls_include_system_certs"] = tls_include_system_certs
        return object()

    monkeypatch.setattr(pyqwest, "HTTPTransport", _http_transport)

    runtime = TinkerRuntime.connect(
        config=TinkerTrainingConfig(
            base_url="https://example.invalid",
            base_model="Qwen/Qwen3.5-4B",
            project_id="project-test",
            seed=7,
        ),
        tokenizer="tokenizer",
        lora_rank=16,
        run_name="unit-test",
    )

    assert captured["service"]["project_id"] == "project-test"
    assert captured["service"]["base_url"] == "https://example.invalid"
    assert captured["service"]["user_metadata"]["run_name"] == "unit-test"
    assert captured["training"]["base_model"] == "Qwen/Qwen3.5-4B"
    assert captured["training"]["rank"] == 16
    assert captured["training"]["seed"] == 7
    tinker_base_client._default_pyqwest_transport()
    assert captured_transport == {"tls_include_system_certs": True}
    asyncio.run(runtime.close())
    assert service.holder.closed is True


def test_configure_tinker_pyqwest_supports_legacy_transport(monkeypatch) -> None:
    import pyqwest
    import tinker._base_client as tinker_base_client

    captured = {}

    def _legacy_http_transport():
        captured["called"] = True
        return object()

    monkeypatch.setattr(pyqwest, "HTTPTransport", _legacy_http_transport)

    tinker_runtime.configure_tinker_pyqwest_system_certs()
    tinker_base_client._default_pyqwest_transport()

    assert captured == {"called": True}


def test_connect_retries_transient_service_bootstrap_connection_errors(
    monkeypatch,
) -> None:
    import tinker

    service = _Service()
    training_client = _TrainingClient()
    service_attempts = 0
    training_attempts = 0
    sleeps = []
    warnings = []

    def _service_factory(**kwargs):
        nonlocal service_attempts
        del kwargs
        service_attempts += 1
        if service_attempts < 3:
            raise tinker.APIConnectionError(
                request=httpx.Request("GET", "https://example.invalid/api/v1/client/config")
            )
        return service

    def _create_lora(**kwargs):
        nonlocal training_attempts
        del kwargs
        training_attempts += 1
        return training_client

    service.create_lora_training_client = _create_lora
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    monkeypatch.setattr(tinker, "ServiceClient", _service_factory)
    monkeypatch.setattr(tinker_runtime, "configure_tinker_pyqwest_system_certs", lambda: None)
    monkeypatch.setattr(tinker_runtime.time, "sleep", sleeps.append)
    monkeypatch.setattr(
        tinker_runtime,
        "logger",
        type("_Logger", (), {"warning": staticmethod(warnings.append)})(),
    )

    runtime = TinkerRuntime.connect(
        config=TinkerTrainingConfig(
            base_model="Qwen/Qwen3.5-4B",
            service_bootstrap_max_attempts=3,
            service_bootstrap_retry_backoff_s=0.25,
        ),
        tokenizer="tokenizer",
        lora_rank=8,
    )

    assert service_attempts == 3
    assert training_attempts == 1
    assert sleeps == [0.25, 0.5]
    assert len(warnings) == 2
    assert all("before any training client is created" in message for message in warnings)
    asyncio.run(runtime.close())


def test_connect_retry_warning_omits_connection_error_details(monkeypatch) -> None:
    import tinker

    service = _Service()
    service.create_lora_training_client = lambda **_: _TrainingClient()
    attempts = 0
    warnings = []
    secret_marker = "must-not-appear-in-logs"

    def _service_factory(**kwargs):
        nonlocal attempts
        del kwargs
        attempts += 1
        if attempts == 1:
            request = httpx.Request(
                "GET",
                "https://example.invalid/api/v1/client/config",
                headers={"Authorization": f"Bearer {secret_marker}"},
            )
            raise tinker.APIConnectionError(
                request=request,
                message=f"request headers contained {secret_marker}",
            )
        return service

    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    monkeypatch.setattr(tinker, "ServiceClient", _service_factory)
    monkeypatch.setattr(tinker_runtime, "configure_tinker_pyqwest_system_certs", lambda: None)
    monkeypatch.setattr(tinker_runtime.time, "sleep", lambda _: None)
    monkeypatch.setattr(
        tinker_runtime,
        "logger",
        type("_Logger", (), {"warning": staticmethod(warnings.append)})(),
    )

    runtime = TinkerRuntime.connect(
        config=TinkerTrainingConfig(
            base_model="Qwen/Qwen3.5-4B",
            service_bootstrap_max_attempts=2,
            service_bootstrap_retry_backoff_s=0,
        ),
        tokenizer="tokenizer",
        lora_rank=8,
    )

    assert attempts == 2
    assert len(warnings) == 1
    assert secret_marker not in warnings[0]
    assert "Exception details are omitted" in warnings[0]
    asyncio.run(runtime.close())


def test_connect_propagates_non_connection_bootstrap_error_immediately(
    monkeypatch,
) -> None:
    import tinker

    service_attempts = 0
    sleeps = []

    def _service_factory(**kwargs):
        nonlocal service_attempts
        del kwargs
        service_attempts += 1
        raise RuntimeError("invalid bootstrap configuration")

    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    monkeypatch.setattr(tinker, "ServiceClient", _service_factory)
    monkeypatch.setattr(tinker_runtime, "configure_tinker_pyqwest_system_certs", lambda: None)
    monkeypatch.setattr(tinker_runtime.time, "sleep", sleeps.append)

    with pytest.raises(RuntimeError, match="invalid bootstrap configuration"):
        TinkerRuntime.connect(
            config=TinkerTrainingConfig(base_model="Qwen/Qwen3.5-4B"),
            tokenizer="tokenizer",
            lora_rank=8,
        )

    assert service_attempts == 1
    assert sleeps == []


def test_connect_propagates_exhausted_bootstrap_connection_error(
    monkeypatch,
) -> None:
    import tinker

    service_attempts = 0
    sleeps = []
    secret_marker = "must-not-appear-in-exhausted-error"

    def _service_factory(**kwargs):
        nonlocal service_attempts
        del kwargs
        service_attempts += 1
        raise tinker.APIConnectionError(
            request=httpx.Request(
                "GET",
                "https://example.invalid/api/v1/client/config",
                headers={"Authorization": f"Bearer {secret_marker}"},
            ),
            message=f"request headers contained {secret_marker}",
        )

    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    monkeypatch.setattr(tinker, "ServiceClient", _service_factory)
    monkeypatch.setattr(tinker_runtime, "configure_tinker_pyqwest_system_certs", lambda: None)
    monkeypatch.setattr(tinker_runtime.time, "sleep", sleeps.append)

    with pytest.raises(
        RuntimeError,
        match="bootstrap failed after 3 connection attempts",
    ) as exc_info:
        TinkerRuntime.connect(
            config=TinkerTrainingConfig(
                base_model="Qwen/Qwen3.5-4B",
                service_bootstrap_max_attempts=3,
                service_bootstrap_retry_backoff_s=0.25,
            ),
            tokenizer="tokenizer",
            lora_rank=8,
        )

    assert service_attempts == 3
    assert sleeps == [0.25, 0.5]
    assert secret_marker not in str(exc_info.value)
    assert exc_info.value.__suppress_context__ is True


def test_connect_does_not_retry_training_client_connection_error(monkeypatch) -> None:
    import tinker

    service = _Service()
    service_attempts = 0
    training_attempts = 0
    sleeps = []

    def _service_factory(**kwargs):
        nonlocal service_attempts
        del kwargs
        service_attempts += 1
        return service

    def _create_lora(**kwargs):
        nonlocal training_attempts
        del kwargs
        training_attempts += 1
        raise tinker.APIConnectionError(request=httpx.Request("POST", "https://example.invalid/api/v1/models"))

    service.create_lora_training_client = _create_lora
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    monkeypatch.setattr(tinker, "ServiceClient", _service_factory)
    monkeypatch.setattr(tinker_runtime, "configure_tinker_pyqwest_system_certs", lambda: None)
    monkeypatch.setattr(tinker_runtime.time, "sleep", sleeps.append)

    with pytest.raises(tinker.APIConnectionError):
        TinkerRuntime.connect(
            config=TinkerTrainingConfig(base_model="Qwen/Qwen3.5-4B"),
            tokenizer="tokenizer",
            lora_rank=8,
        )

    assert service_attempts == 1
    assert training_attempts == 1
    assert sleeps == []
    assert service.holder.closed is True


@pytest.mark.asyncio
async def test_publication_swaps_client_without_changing_inflight_request() -> None:
    training = _TrainingClient()
    runtime = _runtime(training_client=training)

    first = await runtime.publish_sampler_weights()
    first_task = asyncio.create_task(runtime.sample_async(prompt="old-prompt", sampling_params="params"))
    await asyncio.wait_for(training.samplers[0].started.wait(), timeout=1)

    second = await runtime.publish_sampler_weights()
    second_task = asyncio.create_task(runtime.sample_async(prompt="new-prompt", sampling_params="params"))
    await asyncio.wait_for(training.samplers[1].started.wait(), timeout=1)

    training.samplers[0].finish.set()
    training.samplers[1].finish.set()
    first_result, first_identity = await first_task
    second_result, second_identity = await second_task

    assert first.version == 0
    assert second.version == 1
    assert first_result[0] == "v0"
    assert second_result[0] == "v1"
    assert first_identity == first
    assert second_identity == second
    assert runtime.weight_version == 1
    await runtime.close()


@pytest.mark.asyncio
async def test_publication_can_expose_persistent_model_path_for_agent_workers() -> None:
    service = _Service()
    runtime = _runtime(
        service=service,
        config=TinkerTrainingConfig(publish_sampler_model_path=True),
    )

    identity = await runtime.publish_sampler_weights()

    assert identity.version == 0
    assert identity.model_path is not None
    assert identity.model_path.startswith("tinker://training/sampler_weights/skyrl-sampler-v00000000-")
    assert service.created_model_paths == [identity.model_path]
    assert runtime.current_sampler_identity() == identity
    await runtime.close()


@pytest.mark.asyncio
async def test_sampler_publication_awaits_turn_registry_listener() -> None:
    runtime = _runtime(
        service=_Service(),
        config=TinkerTrainingConfig(publish_sampler_model_path=True),
    )
    seen = []

    async def _listener(identity) -> None:
        await asyncio.sleep(0)
        seen.append(identity)

    runtime.add_sampler_publish_listener(_listener)
    first = await runtime.publish_sampler_weights()
    second = await runtime.publish_sampler_weights()

    assert seen == [first, second]
    await runtime.close()


@pytest.mark.asyncio
async def test_failed_publication_preserves_current_sampler() -> None:
    training = _TrainingClient()
    runtime = _runtime(training_client=training)
    first = await runtime.publish_sampler_weights()
    training.fail_next = True

    with pytest.raises(RuntimeError, match="publication failed"):
        await runtime.publish_sampler_weights()

    assert runtime.weight_version == first.version
    task = asyncio.create_task(runtime.sample_async(prompt="prompt", sampling_params="params"))
    await asyncio.wait_for(training.samplers[0].started.wait(), timeout=1)
    training.samplers[0].finish.set()
    _, identity = await task
    assert identity == first
    await runtime.close()


def test_connect_failure_closes_service(monkeypatch) -> None:
    import tinker

    service = _Service()

    def _fail(**kwargs):
        del kwargs
        raise RuntimeError("model unavailable")

    service.create_lora_training_client = _fail
    monkeypatch.setenv("TINKER_API_KEY", "test-key")
    monkeypatch.setattr(tinker, "ServiceClient", lambda **kwargs: service)

    with pytest.raises(RuntimeError, match="model unavailable"):
        TinkerRuntime.connect(
            config=TinkerTrainingConfig(base_model="Qwen/Qwen3.5-4B"),
            tokenizer="tokenizer",
            lora_rank=8,
        )

    assert service.holder.closed is True


def test_usage_report_tracks_training_optimizer_and_checkpoints() -> None:
    runtime = _runtime(
        config=TinkerTrainingConfig(
            base_model="Qwen/Qwen3.5-4B",
            prefill_price_per_million_tokens=1.0,
            cached_prefill_price_per_million_tokens=0.2,
            sample_price_per_million_tokens=2.0,
            train_price_per_million_tokens=0.737,
        )
    )
    runtime.record_forward_backward(training_tokens=1_000, elapsed_s=2.5, succeeded=True)
    runtime.record_optimizer_step(elapsed_s=0.5, succeeded=True)
    runtime.record_external_samples(
        sampling_requests=3,
        prompt_tokens=600,
        prompt_cache_hit_tokens=500,
        sampled_tokens=150,
        elapsed_s=4.0,
    )
    runtime.record_checkpoint(
        global_step=10,
        checkpoint_name="skyrl-step-10-test",
        provider_path="tinker://run/weights/skyrl-step-10-test",
        elapsed_s=1.25,
    )

    metrics = runtime.usage_metrics()
    assert metrics["tinker/usage/training_tokens_total"] == 1_000
    assert metrics["tinker/usage/optimizer_steps_total"] == 1
    assert metrics["tinker/usage/sampling_requests_total"] == 3
    assert metrics["tinker/usage/prompt_tokens_total"] == 600
    assert metrics["tinker/usage/prompt_cache_hit_tokens_total"] == 500
    assert metrics["tinker/usage/prompt_uncached_tokens_total"] == 100
    assert metrics["tinker/usage/prompt_cache_hit_rate"] == pytest.approx(5 / 6)
    assert metrics["tinker/usage/sampled_tokens_total"] == 150
    assert metrics["tinker/usage/checkpoints_total"] == 1
    assert metrics["tinker/estimated_cost/training_usd"] == pytest.approx(0.000737)
    assert metrics["tinker/estimated_cost/prefill_cached_usd"] == pytest.approx(0.0001)
    assert metrics["tinker/estimated_cost/prefill_uncached_usd"] == pytest.approx(0.0001)
    assert metrics["tinker/estimated_cost/prefill_usd"] == pytest.approx(0.0002)
    assert metrics["tinker/estimated_cost/sampling_usd"] == pytest.approx(0.0003)
    assert metrics["tinker/estimated_cost/token_total_usd"] == pytest.approx(0.001237)
    report = runtime.usage_report()
    assert report["checkpoints"][0]["global_step"] == 10
    assert report["checkpoints"][0]["provider_path"].startswith("tinker://")
    assert report["cumulative_across_resumes"] is True


def test_cost_watchdog_records_usage_then_aborts_external_sampling() -> None:
    runtime = _runtime(
        config=TinkerTrainingConfig(
            base_model="Qwen/Qwen3.5-4B",
            prefill_price_per_million_tokens=1.0,
            sample_price_per_million_tokens=2.0,
            train_price_per_million_tokens=3.0,
            max_estimated_cost_usd=1.0,
        )
    )

    with pytest.raises(RuntimeError, match=r"estimated token cost \$1\.20"):
        runtime.record_external_samples(
            sampling_requests=1,
            prompt_tokens=0,
            prompt_cache_hit_tokens=0,
            sampled_tokens=600_000,
            elapsed_s=1.0,
        )

    metrics = runtime.usage_metrics()
    assert metrics["tinker/usage/sampled_tokens_total"] == 600_000
    assert metrics["tinker/estimated_cost/token_total_usd"] == pytest.approx(1.2)
    assert metrics["tinker/estimated_cost/limit_usd"] == 1.0
    assert metrics["tinker/estimated_cost/remaining_usd"] == 0.0
    assert metrics["tinker/estimated_cost/over_limit_usd"] == pytest.approx(0.2)


def test_cost_watchdog_aborts_after_successful_training_usage() -> None:
    runtime = _runtime(
        config=TinkerTrainingConfig(
            base_model="Qwen/Qwen3.5-4B",
            prefill_price_per_million_tokens=1.0,
            sample_price_per_million_tokens=2.0,
            train_price_per_million_tokens=3.0,
            max_estimated_cost_usd=1.0,
        )
    )

    with pytest.raises(RuntimeError, match="watchdog limit"):
        runtime.record_forward_backward(
            training_tokens=400_000,
            elapsed_s=1.0,
            succeeded=True,
        )

    assert runtime.usage_metrics()["tinker/usage/training_tokens_total"] == 400_000


def test_usage_report_restores_cumulative_metrics() -> None:
    runtime = _runtime(
        config=TinkerTrainingConfig(
            base_model="Qwen/Qwen3.5-4B",
            sample_price_per_million_tokens=1.0,
        )
    )
    runtime.restore_usage_reports(
        [
            {
                "cumulative_across_resumes": True,
                "started_at_utc": "2026-07-29T01:00:00+00:00",
                "training_run_id": "training-run-2",
                "training_run_ids": ["training-run-1", "training-run-2"],
                "price_per_million_tokens": {
                    "prefill_uncached": None,
                    "prefill_cached": None,
                    "sample": 1.0,
                    "train": None,
                },
                "metrics": {
                    "tinker/usage/wall_time_seconds": 30.0,
                    "tinker/usage/sampling_requests_total": 5,
                    "tinker/usage/prompt_cache_hit_tokens_total": 0,
                    "tinker/usage/prompt_cache_unknown_tokens_total": 0,
                    "tinker/usage/sampled_tokens_total": 3_000,
                    "tinker/usage/training_tokens_total": 1_200,
                    "tinker/usage/checkpoint_seconds_total": 4.0,
                },
                "checkpoints": [
                    {"global_step": 1, "provider_path": "tinker://one"},
                    {"global_step": 2, "provider_path": "tinker://two"},
                ],
            }
        ]
    )

    metrics = runtime.usage_metrics()
    assert metrics["tinker/usage/wall_time_seconds"] >= 30.0
    assert metrics["tinker/usage/sampling_requests_total"] == 5
    assert metrics["tinker/usage/sampled_tokens_total"] == 3_000
    assert metrics["tinker/usage/training_tokens_total"] == 1_200
    assert metrics["tinker/usage/checkpoints_total"] == 2
    assert metrics["tinker/usage/checkpoint_seconds_total"] == pytest.approx(4.0)
    assert metrics["tinker/estimated_cost/sampling_usd"] == pytest.approx(0.003)
    report = runtime.usage_report()
    assert report["started_at_utc"] == "2026-07-29T01:00:00+00:00"
    assert report["training_run_ids"][:2] == ["training-run-1", "training-run-2"]


def test_usage_restore_rejects_checkpoint_price_mismatch_before_mutating() -> None:
    runtime = _runtime(
        config=TinkerTrainingConfig(
            base_model="Qwen/Qwen3.5-4B",
            prefill_price_per_million_tokens=1.0,
            cached_prefill_price_per_million_tokens=0.2,
            sample_price_per_million_tokens=2.0,
            train_price_per_million_tokens=3.0,
        )
    )

    with pytest.raises(RuntimeError, match="sample"):
        runtime.restore_usage_reports(
            [
                {
                    "cumulative_across_resumes": True,
                    "price_per_million_tokens": {
                        "prefill_uncached": 1.0,
                        "prefill_cached": 0.2,
                        "sample": 9.0,
                        "train": 3.0,
                    },
                    "metrics": {
                        "tinker/usage/sampled_tokens_total": 1_000,
                    },
                }
            ]
        )

    assert runtime.usage_metrics()["tinker/usage/sampled_tokens_total"] == 0


def test_usage_restore_rejects_missing_cache_accounting() -> None:
    runtime = _runtime(
        config=TinkerTrainingConfig(
            base_model="Qwen/Qwen3.5-4B",
            prefill_price_per_million_tokens=1.0,
        )
    )
    with pytest.raises(RuntimeError, match="cache accounting metrics"):
        runtime.restore_usage_reports(
            [
                {
                    "cumulative_across_resumes": True,
                    "price_per_million_tokens": {
                        "prefill_uncached": 1.0,
                        "prefill_cached": 0.2,
                        "sample": None,
                        "train": None,
                    },
                    "metrics": {
                        "tinker/usage/prompt_tokens_total": 1_000,
                    },
                }
            ]
        )

    assert runtime.usage_metrics()["tinker/usage/prompt_tokens_total"] == 0
