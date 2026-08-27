#!/usr/bin/env python3
"""Bounded dense Megatron -> NCCL -> vLLM lifecycle qualification."""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import math
import os
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import ray
import torch
from loguru import logger

# SkyRL must be imported before Megatron modules in the B300 canary image.
from skyrl.train.config import SkyRLTrainConfig, get_config_as_dict
from skyrl.train.entrypoints.main_base import BasePPOExp, validate_cfg
from skyrl.train.trainer import RayPPOTrainer
from skyrl.train.utils import initialize_ray
from skyrl.train.utils.callbacks import TrainingCallback
from skyrl.train.utils.trajectory_logging import TrajectoryLogger
from skyrl_gym.envs import register


RESULT_DIR = Path(os.environ["SKYRL_QUAL_RESULT_DIR"])


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite evidence value: {value}")
        return value
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    return str(value)


def _atomic_json(name: str, value: object) -> None:
    path = RESULT_DIR / name
    temporary = path.with_suffix(path.suffix + ".tmp")
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(_jsonable(value), stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class EvidenceTrajectoryLogger(TrajectoryLogger):
    """Persist complete raw train/eval fingerprints independent of W&B."""

    def __init__(self) -> None:
        super().__init__()
        self._calls: dict[tuple[str, int | None], int] = defaultdict(int)

    def log(self, *, prompts, generator_output, global_step, wandb_key, **kwargs) -> None:
        phase = wandb_key.rsplit("/", 1)[-1]
        call_key = (phase, global_step)
        call_index = self._calls[call_key]
        self._calls[call_key] += 1
        suffix = "" if call_index == 0 else f"-repeat-{call_index}"
        selected = {
            key: generator_output.get(key)
            for key in (
                "prompt_token_ids",
                "response_ids",
                "rollout_logprobs",
                "rewards",
                "loss_masks",
                "stop_reasons",
                "sampler_versions",
            )
        }
        _atomic_json(
            f"trajectory-{phase}-step-{global_step}{suffix}.json",
            {"global_step": global_step, "phase": phase, "prompts": prompts, "output": selected},
        )


class LifecycleEvidenceCallback(TrainingCallback):
    def _version(self, trainer) -> int | None:
        return getattr(trainer.inference_engine_client, "weight_version", None)

    def on_train_start(self, trainer, callback_input, control) -> None:
        # Initial sync is complete when this callback fires. Export the exact
        # trainer weights used for the baseline inference before any update.
        trainer.save_models(checkpoint_step=callback_input.global_step)
        server_urls = list(getattr(trainer.inference_engine_client, "server_urls", []))
        _atomic_json(
            "event-train-start.json",
            {
                "global_step": callback_input.global_step,
                "inference_server_count": len(server_urls),
                "inference_server_urls": server_urls,
                "weight_version": self._version(trainer),
            },
        )

    def on_step_end(self, trainer, callback_input, control) -> None:
        _atomic_json(
            f"event-step-end-{callback_input.global_step}.json",
            {"global_step": callback_input.global_step, "metrics": callback_input.metrics},
        )

    def on_eval_start(self, trainer, callback_input, control) -> None:
        _atomic_json(
            f"event-eval-start-{callback_input.global_step}.json",
            {"global_step": callback_input.global_step, "weight_version": self._version(trainer)},
        )

    def on_save(self, trainer, callback_input, control) -> None:
        _atomic_json(
            f"event-save-{callback_input.global_step}.json",
            {"checkpoint_path": callback_input.ckpt_path, "global_step": callback_input.global_step},
        )

    def on_log(self, trainer, callback_input, control) -> None:
        _atomic_json(
            f"metrics-step-{callback_input.global_step}.json",
            {"global_step": callback_input.global_step, "logs": callback_input.logs},
        )

    def on_train_end(self, trainer, callback_input, control) -> None:
        _atomic_json(
            "event-train-end.json",
            {"global_step": callback_input.global_step, "weight_version": self._version(trainer)},
        )


class TransportCanaryTrainer(RayPPOTrainer):
    """Guarantee a nonzero GRPO signal while retaining real vLLM rollouts."""

    def postprocess_generator_output(self, generator_output, uids, **kwargs):
        original_rewards = list(generator_output["rewards"])
        positions: dict[str, int] = defaultdict(int)
        forced_rewards: list[float] = []
        for uid in uids:
            forced_rewards.append(float(positions[uid] % 2))
            positions[uid] += 1
        if any(count < 2 for count in positions.values()):
            raise RuntimeError(f"transport canary requires at least two samples per prompt: {dict(positions)}")
        generator_output["rewards"] = forced_rewards
        _atomic_json(
            f"transport-rewards-step-{self.global_step}.json",
            {
                "forced_pattern": "alternating-zero-one-within-prompt",
                "original_rewards": original_rewards,
                "forced_rewards": forced_rewards,
                "uids": uids,
            },
        )
        return super().postprocess_generator_output(generator_output, uids, **kwargs)

    async def eval(self, vllm_metrics_scraper=None):
        metrics = await super().eval(vllm_metrics_scraper=vllm_metrics_scraper)
        repeat_step = os.environ.get("SKYRL_QUAL_REPEAT_EVAL_STEP", "0")
        if repeat_step == str(self.global_step):
            # Repeat the selected pinned greedy evaluation with unchanged
            # weights to measure the platform's inference noise floor. Preserve
            # the first built-in eval dump by disabling only that dump.
            dump_eval_results = self.cfg.trainer.dump_eval_results
            self.cfg.trainer.dump_eval_results = False
            try:
                await super().eval(vllm_metrics_scraper=vllm_metrics_scraper)
            finally:
                self.cfg.trainer.dump_eval_results = dump_eval_results
        return metrics


class LifecycleExp(BasePPOExp):
    def get_trainer(
        self,
        cfg,
        tracker,
        tokenizer,
        train_dataset,
        eval_dataset,
        inference_engine_client,
        generator,
        colocate_pg,
    ) -> RayPPOTrainer:
        trainer = TransportCanaryTrainer(
            cfg=cfg,
            tracker=tracker,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            inference_engine_client=inference_engine_client,
            generator=generator,
            colocate_pg=colocate_pg,
        )
        trainer.add_callback(LifecycleEvidenceCallback())
        return trainer

    def get_trajectory_logger(self) -> TrajectoryLogger:
        return EvidenceTrajectoryLogger()

    def _teardown_owned_components(self) -> None:
        logger.info("Qualification teardown: closing all run-owned clients, routers, servers, and trainer actors")
        errors: list[str] = []

        def attempt(label: str, function) -> None:
            try:
                function()
            except BaseException as exc:
                errors.append(f"{label}: {type(exc).__name__}: {exc}")
                logger.exception(f"Qualification teardown failed for {label}")

        if self.trainer is not None and self.trainer.inference_engine_client is not None:
            attempt("inference-client", lambda: asyncio.run(self.trainer.inference_engine_client.teardown()))
        if self._inference_router is not None:
            attempt("inference-router", self._inference_router.shutdown)

        groups = (self._server_groups or []) + (self._prefill_server_groups or []) + (self._decode_server_groups or [])
        unique_groups = {id(group): group for group in groups}
        for index, group in enumerate(unique_groups.values()):
            attempt(f"inference-server-group-{index}", group.shutdown)

        if self.trainer is not None and self.trainer.dispatch is not None:
            actor_groups = getattr(self.trainer.dispatch, "_actor_groups", {})
            actor_handles = {
                str(actor): actor
                for group in actor_groups.values()
                for actor in getattr(group, "_actor_handlers", [])
            }
            for index, actor in enumerate(actor_handles.values()):
                attempt(f"trainer-actor-{index}", lambda actor=actor: ray.kill(actor))

        if errors:
            raise RuntimeError("; ".join(errors))

    def run(self) -> None:
        outcome: dict[str, Any] = {"status": "ERROR"}
        try:
            super().run()
            outcome = {"status": "TRAINING_COMPLETE"}
        except BaseException as exc:
            outcome = {
                "error": f"{type(exc).__name__}: {exc}",
                "status": "ERROR",
                "traceback": traceback.format_exc(),
            }
            raise
        finally:
            try:
                self._teardown_owned_components()
                outcome["owned_component_teardown"] = "complete"
            except BaseException as exc:
                outcome["owned_component_teardown"] = f"error: {type(exc).__name__}: {exc}"
                if outcome["status"] == "TRAINING_COMPLETE":
                    outcome["status"] = "ERROR"
                    _atomic_json("lifecycle-outcome.json", outcome)
                    raise
            _atomic_json("lifecycle-outcome.json", outcome)


@ray.remote(num_cpus=1)
def skyrl_entrypoint(cfg: SkyRLTrainConfig) -> None:
    register(id="multiply", entry_point="examples.train.multiply.env:MultiplyEnv")
    LifecycleExp(cfg).run()


def main() -> None:
    cfg = SkyRLTrainConfig.from_cli_overrides(sys.argv[1:])
    validate_cfg(cfg)
    if os.environ.get("SKYRL_CONFIG_PREFLIGHT_ONLY") == "1":
        _atomic_json("config-preflight.json", {"config": get_config_as_dict(cfg), "status": "PASS"})
        return

    packages = {}
    for package in (
        "flash-attn",
        "megatron-core",
        "ray",
        "torch",
        "transformer-engine",
        "transformers",
        "vllm",
    ):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    _atomic_json(
        "runtime-provenance.json",
        {
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())],
            "cuda_runtime": torch.version.cuda,
            "nccl_runtime": torch.cuda.nccl.version(),
            "packages": packages,
            "torch": torch.__version__,
        },
    )
    initialize_ray(cfg)
    try:
        ray.get(skyrl_entrypoint.remote(cfg))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
