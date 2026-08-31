#!/usr/bin/env python3
"""One-GPU reproducer for in-flight cache reset across a weight-sync sleep.

This uses an unchanged model and bypasses NCCL so the only variable is cache
lifecycle. It runs one deterministic completion normally, one across the old
JSON-body reset, and one across the corrected query-parameter reset. The
worker-level sleep preserves weights on CPU while discarding KV/GDN storage.

The script expects a one-node Ray head in the pinned SkyRL APEX container.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

import ray
from mercor_skyrl import registry
from mercor_skyrl.config import load_overrides
from skyrl.train.utils import validate_cfg
from skyrl.train.utils.utils import initialize_ray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=Path)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=512)
    return parser.parse_args()


def read_metrics(url: str) -> dict[str, float]:
    with urllib.request.urlopen(f"{url}/metrics", timeout=10) as response:
        text = response.read().decode("utf-8")

    wanted = {
        "vllm:num_requests_running": 0.0,
        "vllm:num_requests_waiting": 0.0,
        "vllm:generation_tokens_total": 0.0,
    }
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([^ {]+)(?:\{[^}]*\})?\s+([0-9.eE+-]+)$", line)
        if match and match.group(1) in wanted:
            wanted[match.group(1)] += float(match.group(2))
    return wanted


async def observe_inflight_request(
    task: asyncio.Task[dict[str, Any]],
    url: str,
    settle_s: float = 2.0,
) -> dict[str, float]:
    """Prove the client request is still live without relying on stale gauges."""
    # vLLM's running-request Prometheus gauge is updated on its logging cadence,
    # so a warmed 512-token completion can start and finish between gauge
    # observations. The deliberately long, uncached prompt makes this bounded
    # delay reliable; task.done() is the direct in-process liveness check.
    await asyncio.sleep(settle_s)
    if task.done():
        task.result()  # Surface an HTTP/model exception if that ended it early.
        raise RuntimeError("completion finished before the cache intervention")
    metrics = await asyncio.to_thread(read_metrics, url)
    metrics["client_task_in_flight"] = 1.0
    return metrics


def response_fingerprint(response: dict[str, Any], expected_phrase: str) -> dict[str, Any]:
    choice = response["choices"][0]
    text = choice["text"]
    expected_phrase_count = text.count(expected_phrase)
    nonempty_lines = [line for line in text.splitlines() if line]
    valid_pattern_lines = [line for line in nonempty_lines if expected_phrase.startswith(line)]
    nonnewline_chars = sum(len(line) for line in nonempty_lines)
    valid_pattern_chars = sum(len(line) for line in valid_pattern_lines)
    return {
        "completion_tokens": int(response.get("usage", {}).get("completion_tokens", 0)),
        "finish_reason": choice.get("finish_reason"),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "text_chars": len(text),
        "file_sep_count": text.count("<|file_sep|>"),
        "fim_middle_count": text.count("<|fim_middle|>"),
        "replacement_char_count": text.count("\ufffd"),
        "expected_phrase_count": expected_phrase_count,
        "expected_phrase_char_coverage": round(
            expected_phrase_count * len(expected_phrase) / max(len(text), 1),
            6,
        ),
        "pattern_language_valid": len(valid_pattern_lines) == len(nonempty_lines),
        "pattern_char_compliance": round(valid_pattern_chars / max(nonnewline_chars, 1), 6),
        "nonempty_line_count": len(nonempty_lines),
        "valid_pattern_line_count": len(valid_pattern_lines),
        "head": text[:240],
        "tail": text[-240:],
    }


async def allocator_sleep_preserve_weights_discard_cache(client) -> dict[str, Any]:
    """Bypass EngineCore.sleep so KEEP-paused requests remain logically active."""
    return await client._call_all_servers(
        "/collective_rpc",
        {"method": "sleep", "kwargs": {"level": 1}},
    )


async def allocator_wake(client, tags: list[str]) -> dict[str, Any]:
    return await client._call_all_servers(
        "/collective_rpc",
        {"method": "wake_up", "kwargs": {"tags": tags}},
    )


async def run_completion(client, payload: dict[str, Any]) -> dict[str, Any]:
    return await client.completion({"json": payload})


async def run_crossing_case(
    client,
    payload: dict[str, Any],
    *,
    reset_contract: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    task = asyncio.create_task(run_completion(client, payload))
    running_metrics = await observe_inflight_request(task, client.server_urls[0])

    await client.pause_generation()
    try:
        sleep_response = await allocator_sleep_preserve_weights_discard_cache(client)
        wake_weights_response = await allocator_wake(client, ["weights"])

        if reset_contract == "old-json-body":
            reset_response = await client._call_all_servers(
                "/reset_prefix_cache",
                {"reset_running_requests": True},
            )
        elif reset_contract == "fixed-query":
            reset_response = await client.reset_prefix_cache(reset_running_requests=True)
        else:
            raise ValueError(f"unknown reset contract: {reset_contract}")

        wake_cache_response = await allocator_wake(client, ["kv_cache"])
    except Exception:
        # Best-effort restoration for diagnostic failures. The enclosing job
        # still fails, and production code intentionally does not resume after
        # a failed or partial sync.
        await allocator_wake(client, ["weights", "kv_cache"])
        raise
    else:
        await client.resume_generation()

    response = await asyncio.wait_for(task, timeout=600)
    control = {
        "reset_contract": reset_contract,
        "running_metrics_before_pause": running_metrics,
        "sleep_response": sleep_response,
        "wake_weights_response": wake_weights_response,
        "reset_response": reset_response,
        "wake_cache_response": wake_cache_response,
    }
    return response, control


async def exercise(client, max_tokens: int) -> dict[str, Any]:
    sentinel = "CACHE_RESET_SENTINEL_0123456789"
    prompt = "Continue the exact line pattern below without commentary or variation.\n" + f"{sentinel}\n" * 512
    base_payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": 1234,
        "ignore_eos": True,
        "logprobs": 1,
    }

    # Start with no reusable blocks and give each case a disjoint cache-salt
    # namespace. This avoids a prior case's prefix blocks becoming an input to
    # the next case while preserving identical model/token inputs.
    await client.reset_prefix_cache(reset_running_requests=True)
    baseline = await run_completion(
        client,
        {
            **base_payload,
            "session_id": "hybrid-reset-baseline",
            "cache_salt": "hybrid-reset-baseline-v0",
        },
    )
    old_response, old_control = await run_crossing_case(
        client,
        {
            **base_payload,
            "session_id": "hybrid-reset-old-json",
            "cache_salt": "hybrid-reset-old-json-v0",
        },
        reset_contract="old-json-body",
    )
    fixed_response, fixed_control = await run_crossing_case(
        client,
        {
            **base_payload,
            "session_id": "hybrid-reset-fixed-query",
            "cache_salt": "hybrid-reset-fixed-query-v0",
        },
        reset_contract="fixed-query",
    )

    baseline_text = baseline["choices"][0]["text"]
    old_text = old_response["choices"][0]["text"]
    fixed_text = fixed_response["choices"][0]["text"]
    old_reset_failed = all(
        response.get("body", {}).get("success") is False for response in old_control["reset_response"].values()
    )

    baseline_fingerprint = response_fingerprint(baseline, sentinel)
    old_fingerprint = response_fingerprint(old_response, sentinel)
    fixed_fingerprint = response_fingerprint(fixed_response, sentinel)
    fixed_reset_succeeded = all(
        response.get("body", {}).get("success") is True for response in fixed_control["reset_response"].values()
    )

    # Exact output equality is recorded, but it is not a correctness gate:
    # after forced preemption the hybrid recurrent model is fully re-prefilled,
    # and numerically equivalent CUDA execution can choose a nearby greedy token
    # at a line boundary. The deliberately repeated phrase gives us a stronger
    # semantic invariant: the uninterrupted output must establish a measurable
    # repeated-pattern signal, the broken path must lose most of that signal,
    # and the repaired path must preserve it. Qwen may insert a bounded <think>
    # passage even in the uninterrupted control, so requiring a pure sentinel
    # language would test instruction-following rather than cache correctness.
    baseline_pattern_ok = (
        baseline_fingerprint["expected_phrase_count"] >= 4 and baseline_fingerprint["pattern_char_compliance"] >= 0.10
    )
    old_pattern_corrupted = (
        old_fingerprint["pattern_char_compliance"] < baseline_fingerprint["pattern_char_compliance"] * 0.30
    )
    fixed_pattern_preserved = (
        fixed_fingerprint["expected_phrase_count"] >= baseline_fingerprint["expected_phrase_count"] * 0.90
        and fixed_fingerprint["pattern_char_compliance"] >= baseline_fingerprint["pattern_char_compliance"] * 0.90
        and fixed_fingerprint["completion_tokens"] == baseline_fingerprint["completion_tokens"]
        and fixed_fingerprint["finish_reason"] == baseline_fingerprint["finish_reason"]
    )

    return {
        "baseline": baseline_fingerprint,
        "old_json_body": old_fingerprint,
        "fixed_query": fixed_fingerprint,
        "old_matches_baseline": old_text == baseline_text,
        "fixed_matches_baseline": fixed_text == baseline_text,
        "old_reset_reported_failure": old_reset_failed,
        "fixed_reset_reported_success": fixed_reset_succeeded,
        "baseline_pattern_ok": baseline_pattern_ok,
        "old_pattern_corrupted": old_pattern_corrupted,
        "fixed_pattern_preserved": fixed_pattern_preserved,
        "old_control_plane": old_control,
        "fixed_control_plane": fixed_control,
        "full_responses": {
            "baseline": baseline,
            "old_json_body": old_response,
            "fixed_query": fixed_response,
        },
    }


def main() -> None:
    args = parse_args()
    recipe_cls = registry.get("apex")
    config_cls = recipe_cls.config_cls
    overrides = [
        *load_overrides(args.config),
        f"trainer.run_name={args.run_name}",
        "trainer.log_path=/tmp/skyrl-logs",
        "trainer.logger=none",
        "generator.inference_engine.num_engines=1",
        "generator.inference_engine.tensor_parallel_size=1",
        "generator.inference_engine.distributed_executor_backend=mp",
        "generator.inference_engine.enable_prefix_caching=true",
        "generator.inference_engine.offload_kv_for_weight_sync=true",
    ]
    cfg = config_cls.from_cli_overrides(overrides)
    recipe_cls.prepare_config(cfg, args.config)
    validate_cfg(cfg)
    initialize_ray(cfg)

    experiment = recipe_cls(cfg)
    client = None
    started = time.monotonic()
    try:
        client = experiment.get_inference_client()
        result = asyncio.run(exercise(client, args.max_tokens))
        result.update(
            {
                "elapsed_seconds": round(time.monotonic() - started, 3),
                "model": cfg.trainer.policy.model.path,
                "server_urls": list(client.server_urls),
                "status": (
                    "PASS"
                    if result["old_reset_reported_failure"]
                    and result["fixed_reset_reported_success"]
                    and result["baseline_pattern_ok"]
                    and result["old_pattern_corrupted"]
                    and result["fixed_pattern_preserved"]
                    else "FAIL"
                ),
            }
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        summary = {key: value for key, value in result.items() if key != "full_responses"}
        print("HYBRID_CACHE_RESET_REPRO=" + json.dumps(summary, sort_keys=True), flush=True)
        if result["status"] != "PASS":
            raise RuntimeError("hybrid cache-reset reproducer did not meet parity gates")
    finally:
        if client is not None:
            asyncio.run(client.teardown())
        if experiment._inference_router is not None:
            experiment._inference_router.shutdown()
        for group in experiment._server_groups or []:
            group.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    main()
