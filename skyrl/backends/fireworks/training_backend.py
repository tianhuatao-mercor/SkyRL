"""Trainer-facing dispatch adapter for Fireworks hosted GRPO."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from typing import Any, Optional

from loguru import logger

from skyrl.backends.fireworks.dppo import build_tinker_binary_tv_dppo_request
from skyrl.backends.fireworks.grpo import build_tinker_grpo_datums
from skyrl.backends.fireworks.runtime import FireworksRuntime
from skyrl.backends.skyrl_train.distributed.dispatch import WorkerOutput
from skyrl.backends.skyrl_train.training_batch import TrainingInputBatch
from skyrl.backends.skyrl_train.utils.io import io
from skyrl.train.config import SkyRLTrainConfig


class FireworksPolicyDispatch:
    """Policy-only subset of ``WorkerDispatch`` backed by Fireworks APIs."""

    def __init__(
        self,
        cfg: SkyRLTrainConfig,
        runtime: FireworksRuntime,
        *,
        datum_builder=build_tinker_grpo_datums,
        dppo_request_builder=build_tinker_binary_tv_dppo_request,
    ) -> None:
        self.cfg = cfg
        self.runtime = runtime
        self._datum_builder = datum_builder
        self._dppo_request_builder = dppo_request_builder
        self._last_optim_metrics: dict[str, float] = {}

    def get_lcm_dp_size(self) -> int:
        return 1

    def dp_size(self, model: str) -> int:
        self._require_policy(model)
        return 1

    @staticmethod
    def _require_policy(model: str) -> None:
        if model != "policy":
            raise NotImplementedError(f"Fireworks GRPO is policy-only, got model={model!r}")

    def stage_data(self, model: str, data: TrainingInputBatch, mini_batch_boundaries):
        self._require_policy(model)
        return [data[start:end] for start, end in mini_batch_boundaries]

    def forward_backward_from_staged(
        self,
        model: str,
        staged_batch: TrainingInputBatch,
        loss_fn: Optional[str] = None,
        loss_fn_config: Optional[dict[str, Any]] = None,
        model_id: Optional[str] = None,
    ) -> WorkerOutput:
        self._require_policy(model)
        if loss_fn is not None or loss_fn_config is not None or model_id is not None:
            raise NotImplementedError("Fireworks native GRPO dispatch does not accept per-call loss/model overrides")
        attention_mask = staged_batch.get("attention_mask")
        training_tokens = 0 if attention_mask is None else int(attention_mask.sum().item())
        started = time.monotonic()
        try:
            policy_loss_type = self.cfg.trainer.algorithm.policy_loss_type
            if policy_loss_type == "rollout_is":
                datums = self._datum_builder(
                    staged_batch,
                    max_seq_len=self.cfg.trainer.fireworks.max_seq_len,
                )
                future = self.runtime.training_client.forward_backward(datums, "importance_sampling")
            elif policy_loss_type == "dppo":
                dppo = self.cfg.trainer.algorithm.dppo
                datums, custom_loss = self._dppo_request_builder(
                    staged_batch,
                    max_seq_len=self.cfg.trainer.fireworks.max_seq_len,
                    delta_low=dppo.delta_low,
                    delta_high=dppo.delta_high,
                )
                future = self.runtime.training_client.forward_backward_custom(
                    datums,
                    custom_loss,
                    loss_type_input="logprobs",
                )
            else:  # guarded by config validation; keep dispatch defensive
                raise ValueError(
                    "Fireworks policy dispatch supports only rollout_is or dppo, " f"got {policy_loss_type!r}"
                )
            result = future.result(timeout=self.cfg.trainer.fireworks.request_timeout_s)
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

    def optim_step(self, model: str, model_id: Optional[str] = None) -> Optional[float]:
        self._require_policy(model)
        if model_id is not None:
            raise NotImplementedError("Fireworks GRPO does not support model_id overrides")
        try:
            import tinker
        except ImportError as exc:
            raise ImportError("Fireworks optimizer construction requires the tinker package") from exc

        optimizer = self.cfg.trainer.policy.optimizer_config
        params = tinker.AdamParams(
            learning_rate=optimizer.lr,
            beta1=optimizer.adam_betas[0],
            beta2=optimizer.adam_betas[1],
            eps=self.cfg.trainer.fireworks.adam_eps,
            weight_decay=optimizer.weight_decay,
            grad_clip_norm=optimizer.max_grad_norm,
        )
        result = self.runtime.training_client.optim_step(params).result(
            timeout=self.cfg.trainer.fireworks.request_timeout_s
        )
        self._last_optim_metrics = {
            key: float(value) for key, value in (getattr(result, "metrics", None) or {}).items()
        }
        for key, value in self._last_optim_metrics.items():
            if "grad_norm" in key:
                return value
        return None

    async def save_weights_for_sampler(self, model_id: Optional[str] = None) -> None:
        if model_id is not None:
            raise NotImplementedError("Fireworks GRPO does not support model_id overrides")
        identity = await self.runtime.publish_sampler_weights()
        logger.info(
            "Published Fireworks sampler weights: version={}, snapshot={}",
            identity.version,
            identity.snapshot_path,
        )

    def init_weight_sync_state(self, inference_engine_client) -> None:
        if getattr(inference_engine_client, "runtime", None) is not self.runtime:
            raise ValueError("Fireworks training and inference adapters must share one runtime")

    def empty_cache(self, model: Optional[str] = None) -> None:
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
        """Fireworks checkpoints complete synchronously before this hook returns."""

        self._require_policy(model)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Fireworks GRPO requires rollout logprobs so the local policy forward is skipped")

    def forward_from_staged(self, *args, **kwargs):
        raise NotImplementedError("Fireworks GRPO requires rollout logprobs so the local policy forward is skipped")

    _CHECKPOINT_MANIFEST = "fireworks_checkpoint.json"
    _CHECKPOINT_FORMAT_VERSION = 2
    _CHECKPOINT_IDENTITY_FIELDS = (
        "base_model",
        "training_shape_id",
        "tokenizer_model",
        "lora_rank",
        "lora_alpha",
    )

    def _checkpoint_identity(self) -> dict[str, str | int | None]:
        """Return the provider method identity required for a safe DCP load."""

        lora_rank = self.cfg.trainer.policy.model.lora.rank
        identity: dict[str, str | int | None] = {
            "base_model": self.cfg.trainer.fireworks.base_model or "",
            "training_shape_id": self.cfg.trainer.fireworks.training_shape_id or "",
            "tokenizer_model": self.cfg.trainer.policy.model.path or "",
            "lora_rank": lora_rank,
            # Alpha has no effect for full-parameter training. Canonicalizing
            # it avoids treating two rank-zero runs as different methods just
            # because their otherwise-unused LoRA defaults differ.
            "lora_alpha": (self.cfg.trainer.policy.model.lora.alpha if lora_rank > 0 else None),
        }
        missing = [field for field in ("base_model", "training_shape_id", "tokenizer_model") if not identity[field]]
        if missing:
            raise ValueError("Cannot identify Fireworks checkpoint training method; missing " + ", ".join(missing))
        return identity

    def _preflight_checkpoint_identity(
        self,
        manifest: dict[str, Any],
        *,
        manifest_path: str,
    ) -> None:
        """Reject an incompatible checkpoint before asking Fireworks to load it.

        Version-1 manifests predate the explicit identity. To keep existing
        checkpoints usable, validate any identity fields that are available
        (including model/shape fields from the usage report), warn about the
        fields that cannot be verified, and continue. Version-2 manifests are
        fail-closed: every identity field must be present and match exactly.
        """

        format_version = manifest.get("format_version")
        saved_identity = manifest.get("training_identity")
        if saved_identity is not None and not isinstance(saved_identity, dict):
            raise ValueError("Fireworks checkpoint training_identity must be an object: " f"{manifest_path}")

        if format_version == 1:
            # Real version-1 checkpoints normally contain this usage report,
            # which lets us still reject base-model and shape mismatches. LoRA
            # and tokenizer identity were not recorded and remain unverifiable.
            legacy_identity: dict[str, Any] = dict(saved_identity or {})
            usage_report = manifest.get("usage_at_checkpoint")
            if isinstance(usage_report, dict):
                for field in ("base_model", "training_shape_id"):
                    value = usage_report.get(field)
                    if value not in (None, ""):
                        legacy_identity.setdefault(field, value)
            saved_identity = legacy_identity
        elif format_version == self._CHECKPOINT_FORMAT_VERSION:
            if not isinstance(saved_identity, dict):
                raise ValueError("Fireworks checkpoint format v2 requires training_identity: " f"{manifest_path}")
            missing = [field for field in self._CHECKPOINT_IDENTITY_FIELDS if field not in saved_identity]
            if missing:
                raise ValueError(
                    "Fireworks checkpoint training_identity is incomplete; missing "
                    f"{', '.join(missing)}: {manifest_path}"
                )
        else:
            raise ValueError("Unsupported Fireworks checkpoint manifest version: " f"{format_version!r}")

        current_identity = self._checkpoint_identity()
        mismatches = {
            field: (saved_identity[field], current_identity[field])
            for field in self._CHECKPOINT_IDENTITY_FIELDS
            if field in saved_identity
            and not (
                field == "lora_alpha" and saved_identity.get("lora_rank") == 0 and current_identity["lora_rank"] == 0
            )
            and saved_identity[field] != current_identity[field]
        }
        if mismatches:
            details = ", ".join(
                f"{field}: checkpoint={saved!r}, current={current!r}" for field, (saved, current) in mismatches.items()
            )
            raise ValueError(
                "Fireworks checkpoint training method does not match the current run " f"({details}): {manifest_path}"
            )

        if format_version == 1:
            unverified = [field for field in self._CHECKPOINT_IDENTITY_FIELDS if field not in saved_identity]
            if unverified:
                logger.warning(
                    "Loading legacy Fireworks checkpoint without verifiable {}. "
                    "Version-1 compatibility permits this load; confirm the original "
                    "training method manually. Manifest: {}",
                    ", ".join(unverified),
                    manifest_path,
                )

    def save_checkpoint(self, model: str, ckpt_dir: str, tokenizer=None) -> None:
        """Save persistent Fireworks DCP state and a small local resume manifest.

        ``save_state`` is deliberately separate from the sampler snapshots used
        for per-step hot-loads: DCP includes both weights and optimizer state.
        SkyRL owns the local trainer/dataloader checkpoint alongside this
        manifest, while Fireworks owns the large model checkpoint remotely.
        """

        self._require_policy(model)
        del tokenizer  # Tokenizer files are unchanged and already identified by the config.

        step_match = re.search(r"global_step_(\d+)", ckpt_dir)
        step = step_match.group(1) if step_match else "unknown"
        step_number = int(step) if step.isdigit() else None
        training_identity = self._checkpoint_identity()
        checkpoint_name = f"skyrl-step-{step}-{uuid.uuid4().hex[:8]}"
        result = self.runtime.training_client.save_state(checkpoint_name).result(
            timeout=self.cfg.trainer.fireworks.request_timeout_s
        )
        provider_path = str(getattr(result, "path", "") or "")
        if not provider_path:
            raise RuntimeError(f"Fireworks save_state({checkpoint_name!r}) returned no checkpoint path")

        manifest = {
            "format_version": self._CHECKPOINT_FORMAT_VERSION,
            "checkpoint_kind": "fireworks_dcp",
            "training_identity": training_identity,
            "checkpoint_name": checkpoint_name,
            "provider_path": provider_path,
            "source_trainer_job_id": self.runtime.trainer_job_id,
            # Dedicated Fireworks trainer jobs expose DCP checkpoints through
            # the control plane as step-N, even when save_state() is called
            # with a more descriptive client-side label. Cross-job resume must
            # use the control-plane checkpoint ID rather than that label.
            "cross_job_checkpoint_name": (
                f"step-{step_number}" if self.runtime.trainer_job_id and step_number is not None else None
            ),
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
            "Saved Fireworks DCP checkpoint: name={}, path={}, manifest={}",
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
        """Restore a Fireworks DCP checkpoint, including optimizer by default."""

        self._require_policy(model)
        if load_lr_scheduler_states and not load_optimizer_states:
            raise ValueError("Fireworks cannot restore scheduler state without the optimizer state")

        manifest_path = os.path.join(ckpt_dir, self._CHECKPOINT_MANIFEST)
        if not io.exists(manifest_path):
            raise FileNotFoundError(f"Fireworks checkpoint manifest not found: {manifest_path}")
        with io.open_file(manifest_path, "r") as f:
            manifest = json.load(f)
        self._preflight_checkpoint_identity(
            manifest,
            manifest_path=manifest_path,
        )

        checkpoint_name = str(manifest.get("checkpoint_name") or "")
        provider_path = str(manifest.get("provider_path") or "")
        source_job_id = manifest.get("source_trainer_job_id")
        cross_job_checkpoint_name = str(manifest.get("cross_job_checkpoint_name") or "")
        if not cross_job_checkpoint_name and source_job_id:
            # Compatibility with manifests written before the canonical
            # cross-job name was recorded. Fireworks' dedicated control plane
            # lists durable DCP checkpoints as step-N.
            global_step = manifest.get("global_step")
            if isinstance(global_step, int) and global_step >= 0:
                cross_job_checkpoint_name = f"step-{global_step}"
            else:
                cross_job_checkpoint_name = checkpoint_name
        if source_job_id and cross_job_checkpoint_name:
            load_path = self.runtime.training_client.resolve_checkpoint_path(
                cross_job_checkpoint_name,
                source_job_id=str(source_job_id),
            )
        elif provider_path:
            # Serverless sessions may not expose a dedicated trainer job ID;
            # their returned Tinker path is already directly loadable.
            load_path = provider_path
        else:
            raise ValueError(f"Fireworks checkpoint manifest has no loadable reference: {manifest_path}")

        load = (
            self.runtime.training_client.load_state_with_optimizer
            if load_optimizer_states
            else self.runtime.training_client.load_state
        )
        load(load_path).result(timeout=self.cfg.trainer.fireworks.request_timeout_s)
        restore_usage = getattr(self.runtime, "restore_usage_reports", None)
        usage_report = manifest.get("usage_at_checkpoint")
        if restore_usage is not None and isinstance(usage_report, dict):
            restore_usage([usage_report])
            logger.info("Restored cumulative Fireworks usage from checkpoint report")
        logger.info(
            "Loaded Fireworks DCP checkpoint: reference={}, optimizer_restored={}",
            load_path,
            load_optimizer_states,
        )

    def save_hf_model(self, *args, **kwargs) -> None:
        raise NotImplementedError("Promoting a Fireworks adapter is not implemented yet")
