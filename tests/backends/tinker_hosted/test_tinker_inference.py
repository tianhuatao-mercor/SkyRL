from types import SimpleNamespace

import pytest

from skyrl.backends.tinker.inference import TinkerInferenceClient
from skyrl.backends.tinker.runtime import TinkerRuntime
from skyrl.train.config import TinkerTrainingConfig


class _Holder:
    def close(self):
        return None


class _Service:
    holder = _Holder()


class _Tokenizer:
    def decode(self, tokens, skip_special_tokens=True):
        assert skip_special_tokens is True
        return ":".join(str(token) for token in tokens)


class _Sampler:
    _sampling_session_id = "sampling-session"

    async def sample_async(self, *, prompt, num_samples, sampling_params):
        assert num_samples == 1
        del prompt, sampling_params
        return SimpleNamespace(
            prompt_cache_hit_tokens=1,
            sequences=[
                SimpleNamespace(
                    tokens=[7, 8],
                    logprobs=[-0.7, -0.8],
                    stop_reason="stop",
                )
            ],
        )


class _TrainingClient:
    async def save_weights_and_get_sampling_client_async(self):
        return _Sampler()


@pytest.mark.asyncio
async def test_generate_returns_tokens_logprobs_and_exact_version() -> None:
    runtime = TinkerRuntime(
        service=_Service(),
        training_client=_TrainingClient(),
        tokenizer=_Tokenizer(),
        config=TinkerTrainingConfig(
            base_model="Qwen/Qwen3.5-4B",
            prefill_price_per_million_tokens=0.33,
            sample_price_per_million_tokens=1.005,
            train_price_per_million_tokens=0.737,
        ),
    )
    await runtime.publish_sampler_weights()
    client = TinkerInferenceClient(
        runtime=runtime,
        default_sampling_params={
            "max_tokens": 8,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "logprobs": 1,
        },
    )

    output = await client.generate(
        {
            "prompt_token_ids": [[1, 2], [3, 4]],
            "sampling_params": None,
            "session_ids": ["a", "b"],
            "prompts": None,
            "mm_features": None,
            "cache_salt": None,
        }
    )

    assert output["responses"] == ["7:8", "7:8"]
    assert output["response_ids"] == [[7, 8], [7, 8]]
    assert output["response_logprobs"] == [[-0.7, -0.8], [-0.7, -0.8]]
    assert output["stop_reasons"] == ["stop", "stop"]
    assert output["sampler_versions"] == [0, 0]
    assert client.weight_version == 0
    assert client.model_name == "Qwen/Qwen3.5-4B"
    assert client.get_endpoint_url() == "tinker://hosted"
    usage = runtime.usage_metrics()
    assert usage["tinker/usage/sampling_requests_total"] == 2
    assert usage["tinker/usage/prompt_tokens_total"] == 4
    assert usage["tinker/usage/prompt_cache_hit_tokens_total"] == 2
    assert usage["tinker/usage/prompt_uncached_tokens_total"] == 2
    assert usage["tinker/usage/sampled_tokens_total"] == 4
    assert usage["tinker/usage/sampler_publications_total"] == 1
    sampler, identity = runtime.current_sampler()
    assert sampler is runtime._sampler
    assert identity.version == 0
    assert usage["tinker/estimated_cost/token_total_usd"] == pytest.approx(
        (2 * 0.33 + 2 * 0.33 * 0.2 + 4 * 1.005) / 1_000_000
    )
    await client.teardown()


@pytest.mark.asyncio
async def test_generate_requires_published_sampler() -> None:
    runtime = TinkerRuntime(
        service=_Service(),
        training_client=_TrainingClient(),
        tokenizer=_Tokenizer(),
        config=TinkerTrainingConfig(base_model="Qwen/Qwen3.5-4B"),
    )
    client = TinkerInferenceClient(runtime=runtime, default_sampling_params={"max_tokens": 8})

    with pytest.raises(RuntimeError, match="have not been published"):
        await client.generate(
            {
                "prompt_token_ids": [[1]],
                "sampling_params": None,
                "session_ids": None,
                "prompts": None,
                "mm_features": None,
                "cache_salt": None,
            }
        )
    await runtime.close()
