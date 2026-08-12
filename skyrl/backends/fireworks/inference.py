"""SkyRL inference adapter for Fireworks hosted snapshot samplers."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np

from skyrl.backends.fireworks.router_replay import (
    decode_fireworks_routing_matrices,
)
from skyrl.backends.fireworks.runtime import FireworksRuntime
from skyrl.backends.skyrl_train.inference_servers.base import (
    InferenceEngineInput,
    InferenceEngineInterface,
    InferenceEngineOutput,
)


class FireworksInferenceClient(InferenceEngineInterface):
    """Token-in/token-out client over the runtime's current hosted sampler."""

    def __init__(
        self,
        *,
        runtime: FireworksRuntime,
        default_sampling_params: dict[str, Any],
    ):
        self.runtime = runtime
        self._default_sampling_params = dict(default_sampling_params)
        self._admission = threading.Event()
        self._admission.set()

    async def _wait_for_admission(self) -> None:
        if not self._admission.is_set():
            await asyncio.to_thread(self._admission.wait)

    @property
    def model_name(self) -> str:
        return self.runtime.inference_endpoint.model

    @property
    def weight_version(self) -> int:
        return self.runtime.weight_version

    @staticmethod
    def _tinker_sampling_params(values: dict[str, Any]):
        try:
            import tinker
        except ImportError as exc:
            raise ImportError("Fireworks sampling requires the tinker package") from exc

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

    def _router_replay_sampling_kwargs(
        self,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        max_tokens = values.get("max_tokens")
        if max_tokens is None or int(max_tokens) <= 0:
            raise ValueError("Fireworks router replay requires max_tokens > 0")
        top_k = int(values.get("top_k", -1))
        kwargs: dict[str, Any] = {
            "max_tokens": int(max_tokens),
            "temperature": float(values.get("temperature", 1.0)),
            "max_seq_len": self.runtime.config.max_seq_len,
            "top_p": float(values.get("top_p", 1.0)),
            # Fireworks uses zero, rather than SkyRL/vLLM's -1, for disabled
            # top-k filtering.
            "top_k": 0 if top_k < 0 else top_k,
            "logprobs": True,
            "echo": True,
            "include_routing_matrix": True,
        }
        stop = values.get("stop")
        if stop:
            kwargs["stop"] = list(stop)
        if values.get("seed") is not None:
            kwargs["seed"] = int(values["seed"])
        return kwargs

    async def _sample_one_with_router_replay(
        self,
        tokens: list[int],
        *,
        values: dict[str, Any],
    ) -> tuple[str, list[int], str, list[float], np.ndarray]:
        if not tokens:
            raise ValueError("Fireworks router replay requires a non-empty prompt")
        completions, identity = await self.runtime.sample_with_router_replay_async(
            prompt_token_ids=tokens,
            sampling_kwargs=self._router_replay_sampling_kwargs(values),
        )
        if len(completions) != 1:
            raise RuntimeError(
                f"Fireworks sampler returned {len(completions)} completions for a one-sample "
                f"router-replay request at snapshot {identity.snapshot_path}"
            )
        completion = completions[0]
        full_tokens = [int(token) for token in completion.full_tokens]
        prompt_len = int(completion.prompt_len)
        if prompt_len != len(tokens) or full_tokens[:prompt_len] != tokens:
            raise RuntimeError("Fireworks router-replay response did not preserve the token prompt")
        response_ids = full_tokens[prompt_len:]
        if int(completion.completion_len) != len(response_ids):
            raise RuntimeError("Fireworks router-replay completion length is inconsistent")

        raw_logprobs = completion.sampling_logprobs
        if not completion.logprobs_echoed or raw_logprobs is None:
            raise RuntimeError("Fireworks router replay requires echo-aligned sampling logprobs")
        response_start = prompt_len - 1
        response_end = response_start + len(response_ids)
        response_logprobs_raw = list(raw_logprobs[response_start:response_end])
        if len(response_logprobs_raw) != len(response_ids) or any(value is None for value in response_logprobs_raw):
            raise RuntimeError("Fireworks did not return one sampling logprob per generated token")
        response_logprobs = [float(value) for value in response_logprobs_raw]

        encoded_routes = completion.routing_matrices
        expected_route_rows = len(full_tokens) - 1
        if encoded_routes is None or len(encoded_routes) != expected_route_rows:
            actual = 0 if encoded_routes is None else len(encoded_routes)
            raise RuntimeError(
                "Fireworks router-replay response has misaligned routing rows: "
                f"got {actual}, expected {expected_route_rows}"
            )
        routes = decode_fireworks_routing_matrices(encoded_routes)
        self.runtime.record_router_capture(
            routing_rows=routes.shape[0],
            routing_bytes=int(routes.nbytes),
        )
        response = self.runtime.tokenizer.decode(
            response_ids,
            skip_special_tokens=True,
        )
        return (
            response,
            response_ids,
            self._stop_reason(completion.finish_reason),
            response_logprobs,
            routes,
        )

    async def _sample_one(
        self,
        tokens: list[int],
        *,
        params: Any,
        request_logprobs: bool,
    ) -> tuple[str, list[int], str, list[float] | None, None]:
        try:
            import tinker
        except ImportError as exc:
            raise ImportError("Fireworks sampling requires the tinker package") from exc

        prompt = tinker.ModelInput.from_ints(tokens)
        result, identity = await self.runtime.sample_async(
            prompt=prompt,
            sampling_params=params,
        )
        sequences = list(getattr(result, "sequences", []) or [])
        if len(sequences) != 1:
            raise RuntimeError(
                f"Fireworks sampler returned {len(sequences)} sequences for a one-sample request "
                f"at snapshot {identity.snapshot_path}"
            )
        sequence = sequences[0]
        response_ids = [int(token) for token in list(sequence.tokens)]
        raw_logprobs = getattr(sequence, "logprobs", None)
        response_logprobs = None if raw_logprobs is None else [float(value) for value in list(raw_logprobs)]
        if request_logprobs and (response_logprobs is None or len(response_logprobs) != len(response_ids)):
            raise RuntimeError("Fireworks did not return one rollout logprob per generated token")
        response = self.runtime.tokenizer.decode(response_ids, skip_special_tokens=True)
        return (
            response,
            response_ids,
            self._stop_reason(sequence.stop_reason),
            response_logprobs,
            None,
        )

    async def generate(
        self,
        input_batch: InferenceEngineInput,
        model: Optional[str] = None,
    ) -> InferenceEngineOutput:
        del model
        prompt_token_ids = input_batch.get("prompt_token_ids")
        if prompt_token_ids is None or input_batch.get("prompts") is not None:
            raise ValueError("Fireworks hosted generation currently requires prompt_token_ids only")

        await self._wait_for_admission()
        values: dict[str, Any] = dict(self._default_sampling_params)
        sampling_overrides = input_batch.get("sampling_params")
        if sampling_overrides is not None:
            for key, value in sampling_overrides.items():
                values[key] = value
        router_replay = self.runtime.config.enable_router_replay
        request_logprobs = values.get("logprobs") is not None or router_replay
        params = None if router_replay else self._tinker_sampling_params(values)

        rows = await asyncio.gather(
            *(
                (
                    self._sample_one_with_router_replay(
                        list(tokens),
                        values=values,
                    )
                    if router_replay
                    else self._sample_one(
                        list(tokens),
                        params=params,
                        request_logprobs=request_logprobs,
                    )
                )
                for tokens in prompt_token_ids
            )
        )
        response_logprobs_out: list[list[float]] | None = None
        if request_logprobs:
            response_logprobs_out = []
            for row in rows:
                if row[3] is None:
                    raise RuntimeError(
                        "Fireworks sampling was configured for logprobs but returned none"
                    )
                response_logprobs_out.append(row[3])

        rollout_expert_indices_out: list[np.ndarray] | None = None
        if router_replay:
            rollout_expert_indices_out = []
            for row in rows:
                if row[4] is None:
                    raise RuntimeError(
                        "Fireworks router replay was enabled but returned no routes"
                    )
                rollout_expert_indices_out.append(row[4])

        return InferenceEngineOutput(
            responses=[row[0] for row in rows],
            response_ids=[row[1] for row in rows],
            stop_reasons=[row[2] for row in rows],
            response_logprobs=response_logprobs_out,
            prompt_logprobs=None,
            rollout_expert_indices=rollout_expert_indices_out,
            # A Fireworks request may span a sampler hot-load. The version at
            # request admission therefore is not an exact trajectory version,
            # so do not advertise one to the async staleness logic.
            sampler_versions=None,
        )

    def get_endpoint_url(self) -> str:
        return self.runtime.inference_endpoint.api_base

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        del request_payload
        raise NotImplementedError("Fireworks hosted sampling currently uses token-in/token-out generation")

    async def render_chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        del request_payload
        raise NotImplementedError("Fireworks hosted rendering is performed by the SkyRL tokenizer")

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        del request_payload
        raise NotImplementedError("Use token-in/token-out generate() with Fireworks hosted sampling")

    async def wake_up(self, *args: Any, **kwargs: Any):
        return None

    async def sleep(self, *args: Any, **kwargs: Any):
        return None

    async def init_weight_update_communicator(self, init_info):
        raise NotImplementedError("Fireworks owns hosted weight synchronization")

    async def update_named_weights(self, request):
        raise NotImplementedError("Fireworks owns hosted weight synchronization")

    async def teardown(self):
        await self.runtime.close()

    async def reset_prefix_cache(self, reset_running_requests: bool = False):
        del reset_running_requests
        return None

    async def pause_generation(self) -> None:
        self._admission.clear()

    async def resume_generation(self) -> None:
        self._admission.set()

    async def finish_session(self, session_id: str) -> None:
        del session_id
        return None

    async def get_world_size(self) -> Tuple[int, int]:
        return 1, 1
