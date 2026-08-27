#!/usr/bin/env python3
"""Verify a bounded SkyRL resume without mutating its frozen source checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import torch
from safetensors import safe_open
from torch.distributed.checkpoint import FileSystemReader

from verify_artifacts import (
    _aligned_logprob_delta,
    _atomic_json,
    _checkpoint_state_layout,
    _compare_exports,
    _finite_numbers,
    _json,
    _tensor_digest,
    _tensor_index,
    _validated_inference,
)


def _export_fingerprint(model_dir: Path) -> dict[str, object]:
    index = _tensor_index(model_dir)
    digest = hashlib.sha256()
    tensor_bytes = 0
    for name in sorted(index):
        with safe_open(index[name], framework="pt", device="cpu") as handle:
            tensor = handle.get_tensor(name)
        if not torch.isfinite(tensor).all():
            raise RuntimeError(f"non-finite exported tensor: {model_dir}:{name}")
        _tensor_digest(name, tensor, digest)
        tensor_bytes += tensor.numel() * tensor.element_size()
    return {
        "sha256": digest.hexdigest(),
        "tensor_bytes": tensor_bytes,
        "tensor_count": len(index),
    }


def _assert_same_export(source_dir: Path, resumed_dir: Path) -> dict[str, object]:
    source_index = _tensor_index(source_dir)
    resumed_index = _tensor_index(resumed_dir)
    if set(source_index) != set(resumed_index):
        raise RuntimeError("source/resumed export tensor key sets differ")
    for name in sorted(source_index):
        with safe_open(source_index[name], framework="pt", device="cpu") as handle:
            source = handle.get_tensor(name)
        with safe_open(resumed_index[name], framework="pt", device="cpu") as handle:
            resumed = handle.get_tensor(name)
        if source.dtype != resumed.dtype or source.shape != resumed.shape or not torch.equal(source, resumed):
            raise RuntimeError(f"resumed trainer export differs from frozen source: {name}")
    source_summary = _export_fingerprint(source_dir)
    resumed_summary = _export_fingerprint(resumed_dir)
    if source_summary != resumed_summary:
        raise RuntimeError(f"source/resumed export fingerprints differ: {source_summary} != {resumed_summary}")
    return source_summary


def _resume_inference_delta(source_path: Path, after_path: Path, repeat_path: Path) -> dict[str, object]:
    source = _validated_inference(source_path)
    after = _validated_inference(after_path)
    repeat = _validated_inference(repeat_path)
    for candidate, label in ((after, "post-resume update"), (repeat, "post-resume repeat")):
        if source["prompts"] != candidate["prompts"]:
            raise RuntimeError(f"source and {label} prompts differ")
        if source["output"]["prompt_token_ids"] != candidate["output"]["prompt_token_ids"]:
            raise RuntimeError(f"source and {label} prompt token IDs differ")
    if after["output"]["response_ids"] != repeat["output"]["response_ids"]:
        raise RuntimeError("repeated post-resume inference produced different token IDs")
    repeat_noise = _aligned_logprob_delta(after, repeat)
    token_ids_changed = source["output"]["response_ids"] != after["output"]["response_ids"]
    max_logprob_delta = None
    threshold = max(1e-6, repeat_noise * 10.0 + 1e-7)
    if not token_ids_changed:
        max_logprob_delta = _aligned_logprob_delta(source, after)
        if max_logprob_delta <= threshold:
            raise RuntimeError(
                f"post-resume inference change {max_logprob_delta} did not exceed threshold {threshold}"
            )
    return {
        "post_update_repeat_logprob_noise": repeat_noise,
        "response_token_ids_changed": token_ids_changed,
        "max_comparable_logprob_delta": max_logprob_delta,
        "change_threshold": threshold,
        "samples": len(after["output"]["response_ids"]),
    }


def _verify_hash_log(pre_path: Path, post_path: Path, label: str) -> dict[str, object]:
    pre = pre_path.read_text(encoding="utf-8").splitlines()
    post = post_path.read_text(encoding="utf-8").splitlines()
    if not pre or pre != post or any(not line.endswith(": OK") for line in pre):
        raise RuntimeError(f"{label} checksum verification failed or drifted")
    return {"files": len(pre), "pre_post_identical": True}


def _verify_router(log_dir: Path, expected_receivers: int, expected_host: str) -> dict[str, object]:
    router_logs = sorted(log_dir.glob("router-*.log"))
    if len(router_logs) != 1:
        raise RuntimeError(f"expected exactly one router log, found {router_logs}")
    text = router_logs[0].read_text(encoding="utf-8", errors="replace")
    host_pattern = re.escape(expected_host)
    routes = re.findall(rf"worker='(http://{host_pattern}:\d+)' \(index=(\d+)\)", text)
    counts = {
        index: sum(1 for _, observed in routes if observed == index)
        for index in sorted({observed for _, observed in routes})
    }
    expected = {str(index) for index in range(expected_receivers)}
    urls = {url for url, _ in routes}
    if set(counts) != expected or any(count <= 0 for count in counts.values()) or len(urls) != expected_receivers:
        raise RuntimeError(f"router did not exercise every inference engine: counts={counts}, urls={urls}")
    return {
        "log": str(router_logs[0]),
        "requests_per_engine_index": counts,
        "worker_urls": sorted(urls),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--source-result-dir", type=Path, required=True)
    parser.add_argument("--source-export-dir", type=Path, required=True)
    parser.add_argument("--resume-checkpoint", type=Path, required=True)
    parser.add_argument("--source-evidence-pre", type=Path, required=True)
    parser.add_argument("--source-evidence-post", type=Path, required=True)
    parser.add_argument("--source-checkpoint-pre", type=Path, required=True)
    parser.add_argument("--source-checkpoint-post", type=Path, required=True)
    parser.add_argument("--resume-boundary-record", type=Path, required=True)
    parser.add_argument("--attached-log", type=Path, required=True)
    parser.add_argument("--post-gpu-processes", type=Path, required=True)
    parser.add_argument("--post-containers", type=Path, required=True)
    parser.add_argument("--post-processes", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-inference-receivers", type=int, default=2)
    parser.add_argument("--expected-inference-host", default="127.0.0.1")
    parser.add_argument("--inference-log-dir", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        help="write the verification record outside result-dir for an immutable offline audit",
    )
    args = parser.parse_args()
    if args.expected_inference_receivers < 1:
        parser.error("--expected-inference-receivers must be positive")
    if not args.expected_inference_host or ":" in args.expected_inference_host:
        parser.error("--expected-inference-host must be a non-empty IPv4 address or hostname")

    checks: dict[str, object] = {}
    outcome = _json(args.result_dir / "lifecycle-outcome.json")
    if outcome != {"owned_component_teardown": "complete", "status": "TRAINING_COMPLETE"}:
        raise RuntimeError(f"lifecycle did not complete cleanly: {outcome}")
    checks["lifecycle_outcome"] = outcome

    checks["source_evidence_integrity"] = _verify_hash_log(
        args.source_evidence_pre, args.source_evidence_post, "source evidence"
    )
    checks["source_checkpoint_integrity"] = _verify_hash_log(
        args.source_checkpoint_pre, args.source_checkpoint_post, "source checkpoint"
    )
    boundary_record = _json(args.resume_boundary_record)
    expected_boundary = {
        "iterator_finished": False,
        "num_yielded": 1,
        "sampler_iter_yielded": 1,
        "samples_yielded": 4,
        "workaround": "allow-one-empty-restored-outer-iteration-before-step-2",
    }
    if boundary_record != expected_boundary:
        raise RuntimeError(f"unexpected resume boundary workaround evidence: {boundary_record}")
    checks["resume_boundary_workaround"] = boundary_record
    source_roots = (args.source_result_dir.parent, args.resume_checkpoint.parent.parent)
    writable = [str(path) for root in source_roots for path in (root, *root.rglob("*")) if path.stat().st_mode & 0o222]
    if writable:
        raise RuntimeError(f"frozen resume inputs are writable: {writable[:10]}")

    train_start = _json(args.result_dir / "event-train-start.json")
    urls = train_start.get("inference_server_urls")
    if (
        train_start.get("global_step") != 1
        or train_start.get("weight_version") != 1
        or train_start.get("inference_server_count") != args.expected_inference_receivers
        or not isinstance(urls, list)
        or len(set(urls)) != args.expected_inference_receivers
        or any(not url.startswith(f"http://{args.expected_inference_host}:") for url in urls)
    ):
        raise RuntimeError(f"invalid resumed train-start topology: {train_start}")
    eval_start = _json(args.result_dir / "event-eval-start-2.json")
    if eval_start.get("global_step") != 2 or eval_start.get("weight_version") != 2:
        raise RuntimeError(f"unexpected post-resume weight version: {eval_start}")
    checks["resume_cursor"] = {"loaded_step": 1, "trained_step": 2}
    checks["weight_versions"] = {"resumed_initial_sync": 1, "post_update": 2}
    checks["inference_topology"] = {"expected_receivers": args.expected_inference_receivers, "server_urls": urls}

    metrics = _json(args.result_dir / "metrics-step-2.json")
    logs = metrics["logs"]
    if logs.get("trainer/epoch") != 2 or logs.get("trainer/global_step") != 2:
        raise RuntimeError(f"resume did not cross the recorded dataloader boundary: {logs}")
    grad_value = logs.get("policy/grad_norm")
    if not isinstance(grad_value, (int, float)) or not math.isfinite(float(grad_value)) or grad_value <= 0:
        raise RuntimeError(f"invalid exact policy gradient norm: {grad_value}")
    step_end_grad = _json(args.result_dir / "event-step-end-2.json")["metrics"].get("grad_norm")
    if step_end_grad != grad_value or any(not math.isfinite(value) for value in _finite_numbers(metrics)):
        raise RuntimeError("step-end/logged gradient mismatch or non-finite metric")
    timing_keys = (
        "timing/policy_train",
        "timing/train_critic_and_policy",
        "timing/sync_weights",
        "timing/save_checkpoints",
        "timing/save_hf_model",
    )
    for key in timing_keys:
        value = logs.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value <= 0:
            raise RuntimeError(f"missing/non-positive resumed lifecycle timing {key}: {value}")
    checks["gradient_norm"] = {"key": "policy/grad_norm", "value": grad_value}
    checks["lifecycle_timings"] = {key: logs[key] for key in timing_keys}

    reward_record = _json(args.result_dir / "transport-rewards-step-2.json")
    uids = reward_record["uids"]
    forced = reward_record["forced_rewards"]
    original = reward_record["original_rewards"]
    if not uids or not (len(uids) == len(forced) == len(original)):
        raise RuntimeError("transport canary reward arrays are empty or misaligned")
    positions: dict[str, int] = {}
    for uid, reward in zip(uids, forced):
        position = positions.get(uid, 0)
        if float(reward) != float(position % 2):
            raise RuntimeError(f"transport reward pattern mismatch for {uid} at {position}")
        positions[uid] = position + 1
    if any(count < 2 for count in positions.values()):
        raise RuntimeError(f"transport canary has undersampled prompts: {positions}")
    checks["transport_reward_canary"] = {"prompts": len(positions), "samples": len(uids)}

    sync_records = [_json(args.result_dir / f"weight-sync-transfer-{index}.json") for index in (1, 2)]
    for index, record in enumerate(sync_records, 1):
        if (
            record.get("backend") != "nccl"
            or record.get("completed_after_finish_weight_update") is not True
            or record.get("inference_receiver_ranks") != args.expected_inference_receivers
            or record.get("inference_server_count") != args.expected_inference_receivers
            or record.get("world_size") != args.expected_inference_receivers + 1
            or record.get("transfer_index") != index
            or record.get("tensor_count", 0) <= 0
            or record.get("tensor_bytes", 0) <= 0
        ):
            raise RuntimeError(f"invalid resumed weight-sync evidence: {sync_records}")
    if (
        sync_records[0]["tensor_count"] != sync_records[1]["tensor_count"]
        or sync_records[0]["tensor_bytes"] != sync_records[1]["tensor_bytes"]
    ):
        raise RuntimeError(f"resumed transfer accounting differs: {sync_records}")
    checks["weight_sync_transfers"] = sync_records
    checks["router_distribution"] = _verify_router(
        args.inference_log_dir, args.expected_inference_receivers, args.expected_inference_host
    )

    resumed_export = args.export_dir / "global_step_1" / "policy"
    final_export = args.export_dir / "global_step_2" / "policy"
    checks["exact_resumed_export"] = _assert_same_export(args.source_export_dir, resumed_export)
    checks["parameter_delta"] = _compare_exports(resumed_export, final_export)
    checks["inference_delta"] = _resume_inference_delta(
        args.source_result_dir / "trajectory-eval-step-1.json",
        args.result_dir / "trajectory-eval-step-2.json",
        args.result_dir / "trajectory-eval-step-2-repeat-1.json",
    )

    step_dir = args.checkpoint_dir / "global_step_2"
    policy_dir = step_dir / "policy"
    required = [
        policy_dir / ".metadata",
        policy_dir / "common.pt",
        policy_dir / "metadata.json",
        policy_dir / "huggingface" / "config.json",
        step_dir / "data.pt",
        step_dir / "trainer_state.pt",
    ]
    missing = [str(path) for path in required if not path.exists() or (path.is_file() and path.stat().st_size == 0)]
    distcp = list(policy_dir.glob("*.distcp"))
    if missing or not distcp or any(path.stat().st_size == 0 for path in distcp):
        raise RuntimeError(f"resumed checkpoint incomplete: missing={missing}, distcp={distcp}")
    metadata = FileSystemReader(str(policy_dir)).read_metadata()
    common_state = torch.load(policy_dir / "common.pt", map_location="cpu", weights_only=False)
    state_layout = _checkpoint_state_layout(set(metadata.state_dict_metadata), common_state)
    data_state = torch.load(step_dir / "data.pt", map_location="cpu", weights_only=False)
    trainer_state = torch.load(step_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
    if not isinstance(data_state, dict) or not data_state or trainer_state.get("global_step") != 2:
        raise RuntimeError("resumed checkpoint dataloader/trainer state is incomplete")
    trainer_cfg = trainer_state["config"]["trainer"]
    if (
        trainer_cfg["strategy"] != "megatron"
        or trainer_cfg["policy"]["model"]["lora"]["rank"] != 0
        or trainer_cfg["resume_mode"] != "from_path"
        or Path(trainer_cfg["resume_path"]) != args.resume_checkpoint
        or trainer_cfg["epochs"] != 3
        or trainer_cfg["max_training_steps"] != 2
    ):
        raise RuntimeError("resumed checkpoint does not record the pinned dense resume configuration")
    save_event = _json(args.result_dir / "event-save-2.json")
    if Path(save_event["checkpoint_path"]) != step_dir or save_event.get("global_step") != 2:
        raise RuntimeError(f"resumed save callback path mismatch: {save_event}")
    latest = (args.checkpoint_dir / "latest_ckpt_global_step.txt").read_text(encoding="utf-8").strip()
    if latest != "2":
        raise RuntimeError(f"unexpected resumed latest checkpoint step: {latest!r}")
    checks["checkpoint"] = {
        "global_step": 2,
        "required_components": [str(path) for path in required],
        "state_layout": state_layout,
    }

    if args.post_gpu_processes.read_text(encoding="utf-8").strip():
        raise RuntimeError("GPU compute processes remain after resumed shutdown")
    if args.run_id in args.post_containers.read_text(encoding="utf-8", errors="replace"):
        raise RuntimeError("run-owned resume container remains after shutdown")
    post_processes = args.post_processes.read_text(encoding="utf-8", errors="replace").lower()
    orphan_markers = ("raylet", "gcs_server", "ray::", "vllm.entrypoints", "qualification_entrypoint.py")
    found = [marker for marker in orphan_markers if marker in post_processes]
    if found:
        raise RuntimeError(f"run-owned resumed process markers remain: {found}")
    checks["post_shutdown"] = {"container_absent": True, "gpu_compute_processes": 0, "process_markers": []}

    log_text = args.attached_log.read_text(encoding="utf-8", errors="replace")
    required_markers = [
        f"Loading checkpoint from: {args.resume_checkpoint}",
        "Successfully loaded trainer state",
        "Successfully loaded dataloader state",
        "Successfully loaded policy checkpoint",
        "Successfully loaded complete checkpoint state from global_step_1",
        "Successfully saved checkpoint for global_step_2",
        "Successfully saved model weights.",
        "Training done!",
    ]
    missing_markers = [marker for marker in required_markers if marker not in log_text]
    if missing_markers:
        raise RuntimeError(f"required resume log markers missing: {missing_markers}")
    allowed_modelopt_warning = (
        "UserWarning: Failed to import modelopt vllm plugin due to: "
        "AttributeError('/opt/venvs/skyrl-megatron/lib/python3.12/site-packages/"
        "tilelang/lib/libcudart_stub.so: undefined symbol: cudaDeviceReset'). "
        "You may ignore this warning if you do not need this plugin."
    )
    allowed_count = log_text.count(allowed_modelopt_warning)
    if allowed_count > 1:
        raise RuntimeError(f"unexpected repeated ModelOpt vLLM plugin warning: {allowed_count}")
    fatal_markers = ("CUDA out of memory", "undefined symbol", "NCCL error", "Traceback (most recent call last)")
    fatal_lines = [
        line
        for line in log_text.splitlines()
        if any(marker in line for marker in fatal_markers) and allowed_modelopt_warning not in line
    ]
    if fatal_lines:
        raise RuntimeError(f"fatal markers found in resume log: {fatal_lines[:10]}")
    checks["allowed_warnings"] = {"modelopt_vllm_plugin_cuda_stub": allowed_count}
    checks["log_markers"] = required_markers

    result = {"checks": checks, "status": "PASS"}
    _atomic_json(args.output or args.result_dir / "artifact-verification.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
