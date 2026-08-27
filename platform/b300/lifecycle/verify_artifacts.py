#!/usr/bin/env python3
"""Verify lifecycle outputs and produce an evidence-backed result."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Iterator

import torch
from safetensors import safe_open
from torch.distributed.checkpoint import FileSystemReader


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _safetensor_files(model_dir: Path) -> list[Path]:
    files = sorted(model_dir.glob("*.safetensors"))
    if not files:
        raise RuntimeError(f"no safetensors found in {model_dir}")
    return files


def _tensor_index(model_dir: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in _safetensor_files(model_dir):
        with safe_open(path, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in result:
                    raise RuntimeError(f"duplicate tensor {name} in {model_dir}")
                result[name] = path
    return result


def _tensor_digest(name: str, tensor: torch.Tensor, digest: "hashlib._Hash") -> None:
    digest.update(name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")
    digest.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())


def _compare_exports(before_dir: Path, after_dir: Path) -> dict:
    before_index = _tensor_index(before_dir)
    after_index = _tensor_index(after_dir)
    if set(before_index) != set(after_index):
        raise RuntimeError("pre/post exported tensor key sets differ")

    before_digest = hashlib.sha256()
    after_digest = hashlib.sha256()
    changed = 0
    aggregate_l2_sq = 0.0
    max_abs_delta = 0.0
    total_bytes = 0
    total_tensors = 0
    for name in sorted(before_index):
        with safe_open(before_index[name], framework="pt", device="cpu") as before_handle:
            before = before_handle.get_tensor(name)
        with safe_open(after_index[name], framework="pt", device="cpu") as after_handle:
            after = after_handle.get_tensor(name)
        if before.dtype != after.dtype or before.shape != after.shape:
            raise RuntimeError(f"pre/post tensor metadata differs for {name}")
        if not torch.isfinite(before).all() or not torch.isfinite(after).all():
            raise RuntimeError(f"non-finite pre/post tensor found: {name}")
        _tensor_digest(name, before, before_digest)
        _tensor_digest(name, after, after_digest)
        total_bytes += before.numel() * before.element_size()
        total_tensors += 1
        if not torch.equal(before, after):
            changed += 1
            delta = after.float() - before.float()
            if not torch.isfinite(delta).all():
                raise RuntimeError(f"non-finite parameter delta found: {name}")
            aggregate_l2_sq += float(torch.sum(delta * delta, dtype=torch.float64).item())
            max_abs_delta = max(max_abs_delta, float(delta.abs().max().item()))

    result = {
        "after_sha256": after_digest.hexdigest(),
        "aggregate_l2": math.sqrt(aggregate_l2_sq),
        "before_sha256": before_digest.hexdigest(),
        "changed_tensors": changed,
        "max_abs_delta": max_abs_delta,
        "tensor_bytes": total_bytes,
        "tensor_count": total_tensors,
    }
    if (
        result["before_sha256"] == result["after_sha256"]
        or changed == 0
        or not math.isfinite(result["aggregate_l2"])
        or result["aggregate_l2"] <= 0
        or not math.isfinite(max_abs_delta)
        or max_abs_delta <= 0
    ):
        raise RuntimeError(f"trainer parameters did not change: {result}")
    return result


def _finite_numbers(value) -> Iterator[float]:
    if isinstance(value, dict):
        for item in value.values():
            yield from _finite_numbers(item)
    elif isinstance(value, list):
        for item in value:
            yield from _finite_numbers(item)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        yield float(value)


def _validated_inference(path: Path) -> dict:
    record = _json(path)
    output = record["output"]
    prompt_ids = output["prompt_token_ids"]
    response_ids = output["response_ids"]
    logprobs = output["rollout_logprobs"]
    if not record["prompts"] or not prompt_ids or not response_ids or not logprobs:
        raise RuntimeError(f"inference fingerprint is empty: {path}")
    sample_count = len(response_ids)
    if len(prompt_ids) != sample_count or len(logprobs) != sample_count:
        raise RuntimeError(f"inference sample counts differ: {path}")
    for index, (tokens, probabilities) in enumerate(zip(response_ids, logprobs)):
        if not tokens or len(tokens) != len(probabilities):
            raise RuntimeError(f"response/logprob row mismatch at {path}:{index}")
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in probabilities
        ):
            raise RuntimeError(f"invalid response logprob at {path}:{index}")
    return record


def _aligned_logprob_delta(left: dict, right: dict) -> float:
    left_probs = left["output"]["rollout_logprobs"]
    right_probs = right["output"]["rollout_logprobs"]
    if [len(row) for row in left_probs] != [len(row) for row in right_probs]:
        raise RuntimeError("aligned inference logprob shapes differ")
    return max(
        (
            abs(float(right_value) - float(left_value))
            for left_row, right_row in zip(left_probs, right_probs)
            for left_value, right_value in zip(left_row, right_row)
        ),
        default=0.0,
    )


def _inference_delta(before_path: Path, repeat_path: Path, after_path: Path) -> dict:
    before = _validated_inference(before_path)
    repeat = _validated_inference(repeat_path)
    after = _validated_inference(after_path)
    for candidate, label in ((repeat, "baseline repeat"), (after, "post-update")):
        if before["prompts"] != candidate["prompts"]:
            raise RuntimeError(f"baseline and {label} prompts differ")
        if before["output"]["prompt_token_ids"] != candidate["output"]["prompt_token_ids"]:
            raise RuntimeError(f"baseline and {label} prompt token IDs differ")
        if len(before["output"]["response_ids"]) != len(candidate["output"]["response_ids"]):
            raise RuntimeError(f"baseline and {label} sample counts differ")

    if before["output"]["response_ids"] != repeat["output"]["response_ids"]:
        raise RuntimeError("unchanged-weight baseline repeat produced different token IDs")
    noise_floor = _aligned_logprob_delta(before, repeat)

    before_ids = before["output"]["response_ids"]
    after_ids = after["output"]["response_ids"]
    token_ids_changed = before_ids != after_ids
    max_logprob_delta = None
    threshold = max(1e-6, noise_floor * 10.0 + 1e-7)
    if not token_ids_changed:
        max_logprob_delta = _aligned_logprob_delta(before, after)
        if max_logprob_delta <= threshold:
            raise RuntimeError(
                f"post-update inference change {max_logprob_delta} did not exceed threshold {threshold}"
            )
    return {
        "baseline_samples": len(before_ids),
        "baseline_repeat_logprob_noise": noise_floor,
        "change_threshold": threshold,
        "max_comparable_logprob_delta": max_logprob_delta,
        "post_update_samples": len(after_ids),
        "response_token_ids_changed": token_ids_changed,
    }


def _positive_grad_norm(metrics: dict) -> tuple[str, float]:
    matches: list[tuple[str, float]] = []

    def visit(value, prefix=""):
        if isinstance(value, dict):
            for key, item in value.items():
                path = f"{prefix}.{key}" if prefix else str(key)
                if "grad_norm" in str(key).lower() and isinstance(item, (int, float)):
                    matches.append((path, float(item)))
                visit(item, path)

    visit(metrics)
    finite_positive = [(key, value) for key, value in matches if math.isfinite(value) and value > 0]
    if not finite_positive:
        raise RuntimeError(f"no positive finite gradient norm found: {matches}")
    return max(finite_positive, key=lambda item: item[1])


def _checkpoint_state_layout(sharded_keys: set[str], common_state: dict) -> dict[str, object]:
    if not sharded_keys:
        raise RuntimeError("checkpoint sharded state is empty")
    if not isinstance(common_state, dict):
        raise RuntimeError(f"checkpoint common state is not a mapping: {type(common_state).__name__}")

    sharded_roots = {key.split(".", 1)[0] for key in sharded_keys}
    common_roots = set(common_state)
    all_roots = sharded_roots | common_roots
    required_non_model = {"optimizer", "lr_scheduler", "rng"}
    if not required_non_model.issubset(all_roots):
        raise RuntimeError(
            f"checkpoint non-model state is incomplete: expected {required_non_model}, found {all_roots}"
        )

    if "model" in sharded_roots:
        model_layout = "wrapped_model"
    elif {"embedding", "decoder"}.issubset(sharded_roots):
        # Megatron-Core distributed checkpoints store GPT model roots directly
        # when the model is not wrapped in a top-level state-dict key.
        model_layout = "megatron_gpt_split"
    else:
        raise RuntimeError(
            "checkpoint model state is incomplete: expected sharded root 'model' "
            f"or Megatron roots {{'embedding', 'decoder'}}, found {sharded_roots}"
        )

    return {
        "common_top_level": sorted(common_roots),
        "model_layout": model_layout,
        "sharded_top_level": sorted(sharded_roots),
        "top_level": sorted(all_roots),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--export-dir", type=Path, required=True)
    parser.add_argument("--attached-log", type=Path, required=True)
    parser.add_argument("--post-gpu-processes", type=Path, required=True)
    parser.add_argument("--post-containers", type=Path, required=True)
    parser.add_argument("--post-processes", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--expected-inference-receivers", type=int, default=1)
    parser.add_argument("--inference-log-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="write the verification record outside result-dir for an immutable after-the-fact audit",
    )
    args = parser.parse_args()
    if args.expected_inference_receivers < 1:
        parser.error("--expected-inference-receivers must be positive")

    checks: dict[str, object] = {}
    outcome = _json(args.result_dir / "lifecycle-outcome.json")
    if outcome != {"owned_component_teardown": "complete", "status": "TRAINING_COMPLETE"}:
        raise RuntimeError(f"lifecycle did not complete cleanly: {outcome}")
    checks["lifecycle_outcome"] = outcome

    eval_start_0 = _json(args.result_dir / "event-eval-start-0.json")
    eval_start_1 = _json(args.result_dir / "event-eval-start-1.json")
    if eval_start_0["weight_version"] != 1 or eval_start_1["weight_version"] != 2:
        raise RuntimeError(f"unexpected vLLM weight versions: {eval_start_0}, {eval_start_1}")
    checks["weight_versions"] = {"baseline": 1, "post_update": 2}

    train_start = _json(args.result_dir / "event-train-start.json")
    if args.expected_inference_receivers > 1:
        server_urls = train_start.get("inference_server_urls")
        if (
            train_start.get("inference_server_count") != args.expected_inference_receivers
            or not isinstance(server_urls, list)
            or len(set(server_urls)) != args.expected_inference_receivers
            or any(not url.startswith("http://127.0.0.1:") for url in server_urls)
        ):
            raise RuntimeError(f"invalid multi-engine topology evidence: {train_start}")
    checks["inference_topology"] = {
        "expected_receivers": args.expected_inference_receivers,
        "server_urls": train_start.get("inference_server_urls", []),
    }

    metrics = _json(args.result_dir / "metrics-step-1.json")
    logs = metrics["logs"]
    grad_key = "policy/grad_norm"
    grad_value = logs.get(grad_key)
    if not isinstance(grad_value, (int, float)) or not math.isfinite(float(grad_value)) or grad_value <= 0:
        raise RuntimeError(f"invalid exact policy gradient norm: {grad_value}")
    step_end_grad = _json(args.result_dir / "event-step-end-1.json")["metrics"].get("grad_norm")
    if step_end_grad != grad_value:
        raise RuntimeError(f"step-end/logged gradient norm mismatch: {step_end_grad} != {grad_value}")
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
            raise RuntimeError(f"missing/non-positive lifecycle timing {key}: {value}")
    if any(not math.isfinite(value) for value in _finite_numbers(metrics)):
        raise RuntimeError("non-finite metric found")
    checks["gradient_norm"] = {"key": grad_key, "value": grad_value}
    checks["lifecycle_timings"] = {key: logs[key] for key in timing_keys}

    reward_record = _json(args.result_dir / "transport-rewards-step-1.json")
    uids = reward_record["uids"]
    forced_rewards = reward_record["forced_rewards"]
    original_rewards = reward_record["original_rewards"]
    if not uids or not (len(uids) == len(forced_rewards) == len(original_rewards)):
        raise RuntimeError("transport canary reward arrays are empty or misaligned")
    positions: dict[str, int] = {}
    for uid, reward in zip(uids, forced_rewards):
        position = positions.get(uid, 0)
        if float(reward) != float(position % 2):
            raise RuntimeError(f"transport reward pattern mismatch for {uid} at position {position}")
        positions[uid] = position + 1
    if any(count < 2 for count in positions.values()):
        raise RuntimeError(f"transport canary has undersampled prompts: {positions}")
    if any(not math.isfinite(value) for value in _finite_numbers(original_rewards + forced_rewards)):
        raise RuntimeError("transport canary rewards are non-finite")
    checks["transport_reward_canary"] = {"prompts": len(positions), "samples": len(uids)}

    sync_records = [_json(args.result_dir / f"weight-sync-transfer-{index}.json") for index in (1, 2)]
    if any(
        record.get("backend") != "nccl"
        or record.get("completed_after_finish_weight_update") is not True
        or record.get("inference_receiver_ranks", 1) != args.expected_inference_receivers
        or record.get("inference_server_count", 1) != args.expected_inference_receivers
        or record.get("world_size", 2) != args.expected_inference_receivers + 1
        or record.get("transfer_index") != index
        or record.get("tensor_count", 0) <= 0
        or record.get("tensor_bytes", 0) <= 0
        for index, record in enumerate(sync_records, 1)
    ):
        raise RuntimeError(f"invalid weight-sync transfer evidence: {sync_records}")
    if sync_records[0]["tensor_count"] != sync_records[1]["tensor_count"] or sync_records[0]["tensor_bytes"] != sync_records[1]["tensor_bytes"]:
        raise RuntimeError(f"initial/post-update transfer accounting differs: {sync_records}")
    checks["weight_sync_transfers"] = sync_records

    if args.expected_inference_receivers > 1:
        if args.inference_log_dir is None:
            raise RuntimeError("multi-engine verification requires --inference-log-dir")
        router_logs = sorted(args.inference_log_dir.glob("router-*.log"))
        if len(router_logs) != 1:
            raise RuntimeError(f"expected exactly one router log, found {router_logs}")
        router_text = router_logs[0].read_text(encoding="utf-8", errors="replace")
        routes = re.findall(r"worker='(http://127\.0\.0\.1:\d+)' \(index=(\d+)\)", router_text)
        route_counts = {
            index: sum(1 for _, observed_index in routes if observed_index == index)
            for index in sorted({observed_index for _, observed_index in routes})
        }
        expected_indexes = {str(index) for index in range(args.expected_inference_receivers)}
        if set(route_counts) != expected_indexes or any(count <= 0 for count in route_counts.values()):
            raise RuntimeError(f"router did not exercise every inference engine: {route_counts}")
        routed_urls = {url for url, _ in routes}
        if len(routed_urls) != args.expected_inference_receivers:
            raise RuntimeError(f"router URL count differs from expected topology: {routed_urls}")
        checks["router_distribution"] = {
            "log": str(router_logs[0]),
            "requests_per_engine_index": route_counts,
            "worker_urls": sorted(routed_urls),
        }

    checks["parameter_delta"] = _compare_exports(
        args.export_dir / "global_step_0" / "policy",
        args.export_dir / "global_step_1" / "policy",
    )
    checks["inference_delta"] = _inference_delta(
        args.result_dir / "trajectory-eval-step-0.json",
        args.result_dir / "trajectory-eval-step-0-repeat-1.json",
        args.result_dir / "trajectory-eval-step-1.json",
    )

    step_dir = args.checkpoint_dir / "global_step_1"
    policy_dir = step_dir / "policy"
    required = [
        policy_dir / ".metadata",
        policy_dir / "common.pt",
        policy_dir / "metadata.json",
        policy_dir / "huggingface" / "config.json",
        step_dir / "data.pt",
        step_dir / "trainer_state.pt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"checkpoint components missing: {missing}")
    empty = [str(path) for path in required if path.is_file() and path.stat().st_size == 0]
    distcp_files = list(policy_dir.glob("*.distcp"))
    if empty or not distcp_files or any(path.stat().st_size == 0 for path in distcp_files):
        raise RuntimeError(f"checkpoint contains empty/missing shard files: empty={empty}, distcp={distcp_files}")
    metadata = FileSystemReader(str(policy_dir)).read_metadata()
    sharded_keys = set(metadata.state_dict_metadata)
    common_state = torch.load(policy_dir / "common.pt", map_location="cpu", weights_only=False)
    state_layout = _checkpoint_state_layout(sharded_keys, common_state)
    data_state = torch.load(step_dir / "data.pt", map_location="cpu", weights_only=False)
    trainer_state = torch.load(step_dir / "trainer_state.pt", map_location="cpu", weights_only=False)
    if not isinstance(data_state, dict) or not data_state:
        raise RuntimeError("dataloader checkpoint state is empty")
    if trainer_state.get("global_step") != 1:
        raise RuntimeError(f"trainer checkpoint step mismatch: {trainer_state.get('global_step')}")
    trainer_cfg = trainer_state["config"]["trainer"]
    if trainer_cfg["strategy"] != "megatron" or trainer_cfg["policy"]["model"]["lora"]["rank"] != 0:
        raise RuntimeError("trainer checkpoint is not the pinned dense Megatron configuration")
    save_event = _json(args.result_dir / "event-save-1.json")
    if Path(save_event["checkpoint_path"]) != step_dir:
        raise RuntimeError(f"save callback path mismatch: {save_event}")
    latest = (args.checkpoint_dir / "latest_ckpt_global_step.txt").read_text(encoding="utf-8").strip()
    if latest != "1":
        raise RuntimeError(f"unexpected latest checkpoint step: {latest!r}")
    checks["checkpoint"] = {
        "global_step": 1,
        "required_components": [str(path) for path in required],
        "state_layout": state_layout,
    }

    if args.post_gpu_processes.read_text(encoding="utf-8").strip():
        raise RuntimeError("GPU compute processes remain after shutdown")
    post_containers = args.post_containers.read_text(encoding="utf-8", errors="replace")
    if args.run_id in post_containers:
        raise RuntimeError("run-owned container remains after shutdown")
    post_processes = args.post_processes.read_text(encoding="utf-8", errors="replace").lower()
    orphan_markers = ("raylet", "gcs_server", "ray::", "vllm.entrypoints", "qualification_entrypoint.py")
    found_orphans = [marker for marker in orphan_markers if marker in post_processes]
    if found_orphans:
        raise RuntimeError(f"run-owned process markers remain after shutdown: {found_orphans}")
    checks["post_shutdown"] = {"container_absent": True, "gpu_compute_processes": 0, "process_markers": []}

    log_text = args.attached_log.read_text(encoding="utf-8", errors="replace")
    required_markers = [
        "Successfully saved checkpoint for global_step_1",
        "Successfully saved model weights.",
        "Training done!",
    ]
    missing_markers = [marker for marker in required_markers if marker not in log_text]
    if missing_markers:
        raise RuntimeError(f"required lifecycle log markers missing: {missing_markers}")
    fatal_markers = ["CUDA out of memory", "undefined symbol", "NCCL error", "Traceback (most recent call last)"]
    found_fatal = [marker for marker in fatal_markers if marker in log_text]
    if found_fatal:
        raise RuntimeError(f"fatal markers found in log: {found_fatal}")
    checks["log_markers"] = required_markers

    result = {"checks": checks, "status": "PASS"}
    _atomic_json(args.output or args.result_dir / "artifact-verification.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
