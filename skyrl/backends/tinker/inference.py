"""SkyRL inference adapter for hosted Tinker sampling clients."""

from __future__ import annotations

import asyncio
from typing import Any

from skyrl.backends.skyrl_train.inference_servers.base import (
    InferenceEngineInput,
    InferenceEngineInterface,
    InferenceEngineOutput,
)
from skyrl.backends.tinker.runtime import TinkerRuntime


class TinkerInferenceClient(InferenceEngineInterface):
    """Token-in/token-out inference through the runtime's current sampler."""

    def __init__(
        self,
        *,
        runtime: TinkerRuntime,
        default_sampling_params: dict[str, Any],
    ) -> None:
        self.runtime = runtime
        self._default_sampling_params = dict(default_sampling_params)

    @property
    def model_name(self) -> str:
        return self.runtime.model_name

    @property
    def weight_version(self) -> int:
        return self.runtime.weight_version

    @staticmethod
    def _tinker_sampling_params(values: dict[str, Any]):
        try:
            import tinker
        except ImportError as exc:
            raise ImportError("Hosted Tinker sampling requires the tinker package") from exc

        kwargs = {
            "max_tokens": values.get("max_tokens"),
            "temperature": values.get("temperature", 1.0),
            "top_k": values.get("top_k", -1),
            "top_p": values.get("top_p", 1.0),
            "stop": values.get("stop"),
        }
        if values.get("seed") is not None:
            kwargs["seed"] = values["seed"]
        return tinker.SamplingParams(**kwargs)

    @staticmethod
    def _stop_reason(value: Any) -> str:
        raw = getattr(value, "value", value)
        return "stop" if raw is None else str(raw).lower()

    async def _sample_one(
        self,
        tokens: list[int],
        *,
        params: Any,
        request_logprobs: bool,
    ) -> tuple[str, list[int], str, list[float] | None, int]:
        try:
            import tinker
        except ImportError as exc:
            raise ImportError("Hosted Tinker sampling requires the tinker package") from exc

        result, identity = await self.runtime.sample_async(
            prompt=tinker.ModelInput.from_ints(tokens),
            sampling_params=params,
            prompt_token_count=len(tokens),
        )
        sequences = list(getattr(result, "sequences", []) or [])
        if len(sequences) != 1:
            raise RuntimeError(
                f"Tinker returned {len(sequences)} sequences for a one-sample "
                f"request at sampler version {identity.version}"
            )
        sequence = sequences[0]
        response_ids = [int(token) for token in list(sequence.tokens)]
        raw_logprobs = getattr(sequence, "logprobs", None)
        response_logprobs = None if raw_logprobs is None else [float(value) for value in list(raw_logprobs)]
        if request_logprobs and (response_logprobs is None or len(response_logprobs) != len(response_ids)):
            raise RuntimeError("Tinker did not return one rollout logprob per generated token")
        return (
            self.runtime.tokenizer.decode(response_ids, skip_special_tokens=True),
            response_ids,
            self._stop_reason(sequence.stop_reason),
            response_logprobs,
            identity.version,
        )

    async def generate(
        self,
        input_batch: InferenceEngineInput,
        model: str | None = None,
    ) -> InferenceEngineOutput:
        del model
        prompt_token_ids = input_batch.get("prompt_token_ids")
        if prompt_token_ids is None or input_batch.get("prompts") is not None:
            raise ValueError("Hosted Tinker generation currently requires prompt_token_ids only")

        values = dict(self._default_sampling_params)
        if overrides := input_batch.get("sampling_params"):
            values.update(overrides)
        request_logprobs = values.get("logprobs") is not None
        params = self._tinker_sampling_params(values)
        rows = await asyncio.gather(
            *(
                self._sample_one(
                    list(tokens),
                    params=params,
                    request_logprobs=request_logprobs,
                )
                for tokens in prompt_token_ids
            )
        )
        response_logprobs: list[list[float]] | None = None
        if request_logprobs:
            response_logprobs = []
            for row in rows:
                if row[3] is None:  # Guarded in _sample_one; keeps the output type exact.
                    raise RuntimeError("Tinker returned no rollout logprobs")
                response_logprobs.append(row[3])
        return InferenceEngineOutput(
            responses=[row[0] for row in rows],
            response_ids=[row[1] for row in rows],
            stop_reasons=[row[2] for row in rows],
            response_logprobs=response_logprobs,
            prompt_logprobs=None,
            rollout_expert_indices=None,
            sampler_versions=[row[4] for row in rows],
        )

    def get_endpoint_url(self) -> str:
        return self.runtime.endpoint_url

    async def chat_completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        del request_payload
        raise NotImplementedError("Hosted Tinker currently uses token-in/token-out generation")

    async def render_chat_completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        del request_payload
        raise NotImplementedError("Hosted Tinker rendering is performed by the SkyRL tokenizer")

    async def completion(self, request_payload: dict[str, Any]) -> dict[str, Any]:
        del request_payload
        raise NotImplementedError("Use token-in/token-out generate() with hosted Tinker")

    async def wake_up(self, *args: Any, **kwargs: Any):
        return None

    async def sleep(self, *args: Any, **kwargs: Any):
        return None

    async def init_weight_update_communicator(self, init_info):
        raise NotImplementedError("Tinker owns hosted weight synchronization")

    async def update_named_weights(self, request):
        raise NotImplementedError("Tinker owns hosted weight synchronization")

    async def teardown(self):
        await self.runtime.close()

    async def reset_prefix_cache(self, reset_running_requests: bool = False):
        del reset_running_requests

    async def pause_generation(self) -> None:
        return None

    async def resume_generation(self) -> None:
        return None

    async def finish_session(self, session_id: str) -> None:
        del session_id

    async def get_world_size(self) -> tuple[int, int]:
        return 1, 1
