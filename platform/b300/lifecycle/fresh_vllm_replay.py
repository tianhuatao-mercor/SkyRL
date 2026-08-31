#!/usr/bin/env python3
"""Reload the trained lifecycle export in a fresh vLLM process and replay eval."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EXPECTED_EXPORT = Path(
    "/shared/checkpoints/qualifications/"
    "20260826T233354Z-skyrl-lifecycle-nccl-dense-r1/exports/global_step_1/policy"
)
EXPECTED_TRAJECTORY = Path(
    "/shared/environments/b300/qualifications/"
    "20260826T233354Z-skyrl-lifecycle-nccl-dense-r1/results/trajectory-eval-step-1.json"
)
EXPECTED_SERVED_MODEL = "b300-qwen3-0.6b-lifecycle-step1"
LOGPROB_TOLERANCE = 2e-3
WORKER_EXTENSION = (
    "skyrl.backends.skyrl_train.inference_servers."
    "new_inference_worker_wrap.NewInferenceWorkerWrap"
)


class ReplayFailure(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _choose_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _request_json(url: str, *, payload: dict | None = None, timeout: float) -> tuple[int, Any]:
    body = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {} if body is None else {"Content-Type": "application/json"}
    request = urllib.request.Request(url, data=body, headers=headers, method="GET" if body is None else "POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            return int(response.status), json.loads(raw) if raw else None
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise ReplayFailure(f"HTTP {error.code} from {url}: {detail[:2000]}") from error


def _wait_for_health(base_url: str, process: subprocess.Popen, timeout: float) -> float:
    started = time.monotonic()
    last_error = "no response"
    while time.monotonic() - started < timeout:
        if process.poll() is not None:
            raise ReplayFailure(f"fresh vLLM exited during startup with code {process.returncode}")
        try:
            status, _ = _request_json(f"{base_url}/health", timeout=5.0)
            if status == 200:
                return time.monotonic() - started
            last_error = f"health status {status}"
        except (OSError, TimeoutError, ReplayFailure) as error:
            last_error = str(error)
        time.sleep(2.0)
    raise ReplayFailure(f"fresh vLLM was not healthy within {timeout:.0f}s: {last_error}")


def _terminate_group(process: subprocess.Popen, timeout: float) -> int | None:
    if process.poll() is not None:
        return process.returncode
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return process.poll()
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return process.wait(timeout=10)


def _load_expected(path: Path) -> dict:
    record = json.loads(path.read_text(encoding="utf-8"))
    output = record.get("output", {})
    if record.get("global_step") != 1 or record.get("phase") != "eval":
        raise ReplayFailure("expected trajectory is not the lifecycle step-1 eval")
    required = ("prompt_token_ids", "response_ids", "rollout_logprobs", "stop_reasons")
    if any(not isinstance(output.get(key), list) or len(output[key]) != 2 for key in required):
        raise ReplayFailure("expected trajectory does not contain exactly two complete samples")
    for index, (tokens, logprobs) in enumerate(zip(output["response_ids"], output["rollout_logprobs"])):
        if not tokens or len(tokens) != len(logprobs):
            raise ReplayFailure(f"expected response/logprob mismatch at sample {index}")
        if any(not math.isfinite(float(value)) for value in logprobs):
            raise ReplayFailure(f"expected non-finite logprob at sample {index}")
    return record


def _decode_choice(response: Any) -> dict[str, Any]:
    try:
        choice = response["choices"][0]
        token_ids = choice["token_ids"]
        finish_reason = choice["finish_reason"]
        content = choice["logprobs"]["content"]
        logprobs = [float(item["logprob"]) for item in content]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ReplayFailure(f"invalid token-in/token-out response: {response!r}") from error
    if not token_ids or len(token_ids) != len(logprobs):
        raise ReplayFailure("fresh response token/logprob lengths differ")
    if any(not math.isfinite(value) for value in logprobs):
        raise ReplayFailure("fresh response contains a non-finite logprob")
    return {"finish_reason": finish_reason, "logprobs": logprobs, "token_ids": token_ids}


def _compare(expected: dict, actual: list[dict], label: str) -> dict:
    expected_output = expected["output"]
    deltas: list[float] = []
    for index, record in enumerate(actual):
        expected_ids = expected_output["response_ids"][index]
        expected_logprobs = expected_output["rollout_logprobs"][index]
        if record["token_ids"] != expected_ids:
            mismatch = next(
                (
                    position
                    for position, pair in enumerate(zip(record["token_ids"], expected_ids))
                    if pair[0] != pair[1]
                ),
                min(len(record["token_ids"]), len(expected_ids)),
            )
            raise ReplayFailure(f"{label} sample {index} token IDs differ at position {mismatch}")
        if record["finish_reason"] != expected_output["stop_reasons"][index]:
            raise ReplayFailure(f"{label} sample {index} finish reason differs")
        if len(record["logprobs"]) != len(expected_logprobs):
            raise ReplayFailure(f"{label} sample {index} logprob length differs")
        deltas.extend(abs(left - float(right)) for left, right in zip(record["logprobs"], expected_logprobs))
    maximum = max(deltas, default=0.0)
    if maximum > LOGPROB_TOLERANCE:
        raise ReplayFailure(
            f"{label} maximum logprob delta {maximum} exceeds tolerance {LOGPROB_TOLERANCE}"
        )
    return {"exact_response_token_ids": True, "max_abs_logprob_delta": maximum, "samples": len(actual)}


def _generate_pass(base_url: str, expected: dict, pass_index: int, timeout: float) -> list[dict]:
    sampling_params = {
        "logprobs": 1,
        "max_tokens": 128,
        "min_p": 0.0,
        "repetition_penalty": 1.0,
        "temperature": 0.0,
        "top_k": -1,
        "top_p": 1.0,
    }
    results = []
    for sample_index, prompt_ids in enumerate(expected["output"]["prompt_token_ids"]):
        payload = {
            "cache_salt": f"fresh-replay-pass-{pass_index}-sample-{sample_index}",
            "model": EXPECTED_SERVED_MODEL,
            "request_id": f"fresh-replay-{pass_index}-{sample_index}",
            "sampling_params": sampling_params,
            "token_ids": prompt_ids,
        }
        status, response = _request_json(
            f"{base_url}/inference/v1/generate", payload=payload, timeout=timeout
        )
        if status != 200:
            raise ReplayFailure(f"generate returned status {status}")
        results.append(_decode_choice(response))
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--export-path", type=Path, required=True)
    parser.add_argument("--expected-trajectory", type=Path, required=True)
    parser.add_argument("--result-path", type=Path, required=True)
    parser.add_argument("--server-log-path", type=Path, required=True)
    parser.add_argument("--startup-timeout-seconds", type=int, default=600)
    args = parser.parse_args()
    if args.export_path != EXPECTED_EXPORT or args.expected_trajectory != EXPECTED_TRAJECTORY:
        parser.error("export and trajectory must be the exact pinned lifecycle artifacts")
    return args


def main() -> int:
    args = _parse_args()
    result: dict[str, Any] = {
        "finished_at_utc": None,
        "logprob_tolerance": LOGPROB_TOLERANCE,
        "run_id": args.run_id,
        "started_at_utc": _utc_now(),
        "status": "FAIL",
    }
    process = None
    server_log = None
    started = time.monotonic()
    return_code = 1
    try:
        if args.result_path.exists() or args.server_log_path.exists():
            raise ReplayFailure("refusing to overwrite replay result or server log")
        if not (args.export_path / "model.safetensors").is_file():
            raise ReplayFailure("trained export is incomplete")
        expected = _load_expected(args.expected_trajectory)
        result["runtime"] = {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "packages": {
                package: importlib.metadata.version(package)
                for package in ("flash-attn", "torch", "transformers", "vllm")
            },
        }
        port = _choose_port()
        base_url = f"http://127.0.0.1:{port}"
        command = [
            sys.executable,
            "-m",
            "skyrl.backends.skyrl_train.inference_servers.vllm_server_actor",
            "--model",
            str(args.export_path),
            "--served-model-name",
            EXPECTED_SERVED_MODEL,
            "--tensor-parallel-size",
            "1",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--dtype",
            "bfloat16",
            "--enforce-eager",
            "--gpu-memory-utilization",
            "0.15",
            "--max-model-len",
            "1024",
            "--max-num-seqs",
            "4",
            "--max-num-batched-tokens",
            "1024",
            "--seed",
            "42",
            "--trust-remote-code",
            "--worker-extension-cls",
            WORKER_EXTENSION,
        ]
        result["server"] = {"base_url": base_url, "command": command}
        server_log = args.server_log_path.open("xb", buffering=0)
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            start_new_session=True,
        )
        result["server"]["pid"] = process.pid
        result["server"]["startup_seconds"] = _wait_for_health(
            base_url, process, float(args.startup_timeout_seconds)
        )
        first = _generate_pass(base_url, expected, 1, 180.0)
        repeat = _generate_pass(base_url, expected, 2, 180.0)
        result["first_pass"] = {"comparison": _compare(expected, first, "first"), "responses": first}
        result["repeat_pass"] = {
            "comparison": _compare(expected, repeat, "repeat"),
            "responses": repeat,
        }
        if [record["token_ids"] for record in first] != [record["token_ids"] for record in repeat]:
            raise ReplayFailure("fresh replay is not token-deterministic across repeats")
        result["status"] = "PASS"
        print("FRESH_VLLM_REPLAY_PASS", flush=True)
        return_code = 0
    except BaseException as error:
        result["error"] = {
            "message": str(error),
            "traceback": traceback.format_exc(),
            "type": type(error).__name__,
        }
        print(f"FRESH_VLLM_REPLAY_FAIL {type(error).__name__}: {error}", file=sys.stderr, flush=True)
    finally:
        if process is not None:
            result["server_exit_code"] = _terminate_group(process, 30.0)
        if server_log is not None:
            server_log.close()
        result["duration_seconds"] = time.monotonic() - started
        result["finished_at_utc"] = _utc_now()
        try:
            _atomic_json(args.result_path, result)
        except BaseException as error:
            print(f"failed to persist replay result: {error}", file=sys.stderr, flush=True)
            return_code = 1
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
