"""Trainer-facing policy dispatch for hosted Tinker GRPO."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any

from loguru import logger

from skyrl.backends.fireworks.grpo import build_tinker_grpo_datums
from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.io import io
from skyrl.backends.tinker.runtime import TinkerRuntime
from skyrl.train.config import SkyRLTrainConfig


class TinkerPolicyDispatch:
    """Policy-only ``WorkerDispatch`` subset backed by hosted Tinker APIs."""

    def __init__(
        self,
        cfg: SkyRLTrainConfig,
        runtime: TinkerRuntime,
        *,
        datum_builder=build_tinker_grpo_datums,
    ) -> None:
        self.cfg = cfg
        self.runtime = runtime
        self._datum_builder = datum_builder
        self._last_optim_metrics: dict[str, float] = {}

    def get_lcm_dp_size(self) -> int:
        return 1

    def dp_size(self, model: str) -> int:
        self._require_policy(model)
        return 1

    @staticmethod
    def _require_policy(model: str) -> None:
        if model != "policy":
            raise NotImplementedError(f"Hosted Tinker GRPO is policy-only, got model={model!r}")

    def stage_data(self, model: str, data: TrainingInputBatch, mini_batch_boundaries):
        self._require_policy(model)
        return [data[start:end] for start, end in mini_batch_boundaries]

    def forward_backward_from_staged(
        self,
        model: str,
        staged_batch: TrainingInputBatch,
        loss_fn: str | None = None,
        loss_fn_config: dict[str, Any] | None = None,
        model_id: str | None = None,
    ) -> WorkerOutput:
        self._require_policy(model)
        if loss_fn is not None or loss_fn_config is not None or model_id is not None:
            raise NotImplementedError("Hosted Tinker GRPO does not accept per-call loss/model overrides")
        datums = self._datum_builder(
            staged_batch,
            max_seq_len=self.cfg.trainer.tinker.max_seq_len,
        )
        attention_mask = staged_batch.get("attention_mask")
        training_tokens = 0 if attention_mask is None else int(attention_mask.sum().item())
        started = time.monotonic()
        try:
            result = self.runtime.training_client.forward_backward(datums, "importance_sampling").result(
                timeout=self.cfg.trainer.tinker.request_timeout_s
            )
        except BaseException:
            record = getattr(self.runtime, "record_forward_backward", None)
            if record is not None:
                record(
                    training_tokens=0,
                    elapsed_s=time.monotonic() - started,
                    succeeded=False,
                )
            raise
        record = getattr(self.runtime, "record_forward_backward", None)
        if record is not None:
            record(
                training_tokens=training_tokens,
                elapsed_s=time.monotonic() - started,
                succeeded=True,
            )
        metrics = {key: float(value) for key, value in (getattr(result, "metrics", None) or {}).items()}
        if "loss:sum" in metrics:
            metrics.setdefault("final_loss", metrics["loss:sum"])
        return WorkerOutput(
            loss_fn_output_type=str(getattr(result, "loss_fn_output_type", "scalar")),
            loss_fn_outputs=[],
            metrics=metrics,
        )

    def optim_step(self, model: str, model_id: str | None = None) -> float | None:
        self._require_policy(model)
        if model_id is not None:
            raise NotImplementedError("Hosted Tinker GRPO does not support model_id overrides")
        try:
            import tinker
        except ImportError as exc:
            raise ImportError("Hosted Tinker optimizer construction requires the tinker package") from exc

        optimizer = self.cfg.trainer.policy.optimizer_config
        params = tinker.AdamParams(
            learning_rate=optimizer.lr,
            beta1=optimizer.adam_betas[0],
            beta2=optimizer.adam_betas[1],
            eps=self.cfg.trainer.tinker.adam_eps,
            weight_decay=optimizer.weight_decay,
            grad_clip_norm=optimizer.max_grad_norm,
        )
        started = time.monotonic()
        try:
            result = self.runtime.training_client.optim_step(params).result(
                timeout=self.cfg.trainer.tinker.request_timeout_s
            )
        except BaseException:
            record = getattr(self.runtime, "record_optimizer_step", None)
            if record is not None:
                record(elapsed_s=time.monotonic() - started, succeeded=False)
            raise
        record = getattr(self.runtime, "record_optimizer_step", None)
        if record is not None:
            record(elapsed_s=time.monotonic() - started, succeeded=True)
        self._last_optim_metrics = {
            key: float(value) for key, value in (getattr(result, "metrics", None) or {}).items()
        }
        for key, value in self._last_optim_metrics.items():
            if "grad_norm" in key:
                return value
        return None

    async def save_weights_for_sampler(self, model_id: str | None = None) -> None:
        if model_id is not None:
            raise NotImplementedError("Hosted Tinker GRPO does not support model_id overrides")
        identity = await self.runtime.publish_sampler_weights()
        logger.info(
            "Published hosted Tinker sampler weights: version={}, session={}, model_path={}",
            identity.version,
            identity.sampling_session_id or "<opaque>",
            identity.model_path or "<ephemeral>",
        )

    def init_weight_sync_state(self, inference_engine_client) -> None:
        if getattr(inference_engine_client, "runtime", None) is not self.runtime:
            raise ValueError("Tinker training and inference adapters must share one runtime")

    def empty_cache(self, model: str | None = None) -> None:
        if model is not None:
            self._require_policy(model)

    def mark_all_offloaded(self) -> None:
        return None

    def get_node_ids(self) -> list[str]:
        return []

    def start_profile(self, model: str) -> None:
        self._require_policy(model)

    def profile_step(self, model: str) -> None:
        self._require_policy(model)

    def stop_profile(self, model: str) -> None:
        self._require_policy(model)

    def finalize_pending_saves(self, model: str) -> None:
        self._require_policy(model)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Hosted Tinker GRPO requires rollout logprobs, so local policy " "forward is skipped")

    def forward_from_staged(self, *args, **kwargs):
        raise NotImplementedError("Hosted Tinker GRPO requires rollout logprobs, so local policy " "forward is skipped")

    _CHECKPOINT_MANIFEST = "tinker_checkpoint.json"

    def _usage_reports_for_restore(
        self,
        *,
        ckpt_dir: str,
        manifest: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Return the cumulative report, or merge pre-fix process-local reports."""

        latest_report = manifest.get("usage_at_checkpoint")
        if not isinstance(latest_report, dict):
            return []
        if latest_report.get("cumulative_across_resumes") is True:
            return [latest_report]

        step_number = manifest.get("global_step")
        if not isinstance(step_number, int):
            return [latest_report]

        checkpoint_root = os.path.dirname(os.path.dirname(ckpt_dir))
        legacy_reports: list[tuple[int, dict[str, Any]]] = []
        try:
            checkpoint_dirs = io.list_dir(checkpoint_root)
        except (FileNotFoundError, OSError):
            return [latest_report]

        for candidate_dir in checkpoint_dirs:
            match = re.fullmatch(r"global_step_(\d+)", os.path.basename(candidate_dir))
            if match is None:
                continue
            candidate_step = int(match.group(1))
            if candidate_step > step_number:
                continue
            candidate_manifest_path = os.path.join(candidate_dir, "policy", self._CHECKPOINT_MANIFEST)
            if not io.exists(candidate_manifest_path):
                continue
            with io.open_file(candidate_manifest_path, "r") as f:
                candidate_manifest = json.load(f)
            candidate_report = candidate_manifest.get("usage_at_checkpoint")
            if not isinstance(candidate_report, dict):
                continue
            if candidate_report.get("cumulative_across_resumes") is True:
                # A cumulative report already includes every earlier process.
                legacy_reports = [(candidate_step, candidate_report)]
            else:
                legacy_reports.append((candidate_step, candidate_report))

        return [report for _, report in sorted(legacy_reports)] or [latest_report]

    def save_checkpoint(self, model: str, ckpt_dir: str, tokenizer=None) -> None:
        self._require_policy(model)
        del tokenizer

        step_match = re.search(r"global_step_(\d+)", ckpt_dir)
        step = step_match.group(1) if step_match else "unknown"
        checkpoint_name = f"skyrl-step-{step}-{uuid.uuid4().hex[:8]}"
        started = time.monotonic()
        result = self.runtime.training_client.save_state(
            checkpoint_name,
            ttl_seconds=self.cfg.trainer.tinker.checkpoint_ttl_seconds,
        ).result(timeout=self.cfg.trainer.tinker.request_timeout_s)
        provider_path = str(getattr(result, "path", "") or "")
        if not provider_path:
            raise RuntimeError(f"Tinker save_state({checkpoint_name!r}) returned no checkpoint path")

        step_number = int(step) if step.isdigit() else None
        record = getattr(self.runtime, "record_checkpoint", None)
        if record is not None:
            record(
                global_step=step_number,
                checkpoint_name=checkpoint_name,
                provider_path=provider_path,
                elapsed_s=time.monotonic() - started,
            )
        manifest = {
            "format_version": 1,
            "checkpoint_kind": "tinker_state",
            "checkpoint_name": checkpoint_name,
            "provider_path": provider_path,
            "includes_optimizer_state": True,
            "global_step": step_number,
        }
        usage_report = getattr(self.runtime, "usage_report", None)
        if usage_report is not None:
            manifest["usage_at_checkpoint"] = usage_report()
        io.makedirs(ckpt_dir, exist_ok=True)
        manifest_path = os.path.join(ckpt_dir, self._CHECKPOINT_MANIFEST)
        with io.open_file(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
            f.write("\n")
        logger.info(
            "Saved hosted Tinker checkpoint: name={}, path={}, manifest={}",
            checkpoint_name,
            provider_path,
            manifest_path,
        )

    def load_checkpoint(
        self,
        model: str,
        ckpt_dir: str,
        *,
        load_optimizer_states: bool = True,
        load_lr_scheduler_states: bool = True,
    ) -> None:
        self._require_policy(model)
        if load_lr_scheduler_states and not load_optimizer_states:
            raise ValueError("Tinker cannot restore scheduler state without optimizer state")

        manifest_path = os.path.join(ckpt_dir, self._CHECKPOINT_MANIFEST)
        if not io.exists(manifest_path):
            raise FileNotFoundError(f"Tinker checkpoint manifest not found: {manifest_path}")
        with io.open_file(manifest_path, "r") as f:
            manifest = json.load(f)
        if manifest.get("format_version") != 1:
            raise ValueError("Unsupported Tinker checkpoint manifest version: " f"{manifest.get('format_version')!r}")
        provider_path = str(manifest.get("provider_path") or "")
        if not provider_path:
            raise ValueError(f"Tinker checkpoint manifest has no provider_path: {manifest_path}")

        load = (
            self.runtime.training_client.load_state_with_optimizer
            if load_optimizer_states
            else self.runtime.training_client.load_state
        )
        load(provider_path).result(timeout=self.cfg.trainer.tinker.request_timeout_s)
        restore_usage = getattr(self.runtime, "restore_usage_reports", None)
        if restore_usage is not None:
            usage_reports = self._usage_reports_for_restore(
                ckpt_dir=ckpt_dir,
                manifest=manifest,
            )
            restore_usage(usage_reports)
            if usage_reports:
                logger.info(
                    "Restored cumulative Tinker usage from {} checkpoint report(s)",
                    len(usage_reports),
                )
        logger.info(
            "Loaded hosted Tinker checkpoint: path={}, optimizer_restored={}",
            provider_path,
            load_optimizer_states,
        )

    def save_hf_model(self, *args, **kwargs) -> None:
        raise NotImplementedError("Exporting a hosted Tinker adapter to HuggingFace format is not " "implemented yet")
