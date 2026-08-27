#!/usr/bin/env python3
"""Validate and summarize vLLM serving topology benchmark JSON files."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


class ResultError(RuntimeError):
    """A benchmark result is incomplete or inconsistent."""


def load_rows(result_dir: Path, expected_model: str) -> list[dict]:
    rows = []
    for path in sorted(result_dir.glob("*.json")):
        source = json.loads(path.read_text(encoding="utf-8"))
        if source.get("model_family") != expected_model:
            raise ResultError(f"model family mismatch in {path}")
        completed = source.get("completed")
        failed = source.get("failed")
        prompts = source.get("num_prompts")
        if completed != prompts or failed != 0:
            raise ResultError(
                f"incomplete benchmark in {path}: completed={completed}, failed={failed}, prompts={prompts}"
            )
        gpu_count = int(source["gpu_count"])
        if gpu_count < 1 or 8 % gpu_count:
            raise ResultError(f"unsupported GPU count in {path}: {gpu_count}")
        output_throughput = float(source["output_throughput"])
        row = {
            "completed": completed,
            "duration_s": float(source["duration"]),
            "gpu_count": gpu_count,
            "input_tokens": int(source["input_tokens"]),
            "mean_itl_ms": float(source["mean_itl_ms"]),
            "mean_tpot_ms": float(source["mean_tpot_ms"]),
            "mean_ttft_ms": float(source["mean_ttft_ms"]),
            "output_throughput_per_gpu": output_throughput / gpu_count,
            "output_throughput_server": output_throughput,
            "output_tokens": int(source["output_tokens"]),
            "p50_tpot_ms": float(source["p50_tpot_ms"]),
            "p50_ttft_ms": float(source["p50_ttft_ms"]),
            "p90_tpot_ms": float(source["p90_tpot_ms"]),
            "p90_ttft_ms": float(source["p90_ttft_ms"]),
            "p99_tpot_ms": float(source["p99_tpot_ms"]),
            "p99_ttft_ms": float(source["p99_ttft_ms"]),
            "projected_eight_gpu_node_output_throughput": output_throughput * (8 // gpu_count),
            "request_throughput": float(source["request_throughput"]),
            "result_file": path.name,
            "topology": source["topology"],
        }
        if not all(
            math.isfinite(value) and value >= 0
            for key, value in row.items()
            if isinstance(value, float) and key != "duration_s"
        ):
            raise ResultError(f"non-finite or negative metric in {path}")
        rows.append(row)
    if not rows:
        raise ResultError(f"no result JSON files found in {result_dir}")
    return rows


def summarize(rows: list[dict], model: str) -> dict:
    workloads: dict[tuple[int, int], list[dict]] = {}
    for row in rows:
        workloads.setdefault((row["input_tokens"], row["output_tokens"]), []).append(row)
    comparisons = []
    for (input_tokens, output_tokens), workload_rows in sorted(workloads.items()):
        by_concurrency: dict[int, list[dict]] = {}
        for row in workload_rows:
            source_concurrency = int(row["result_file"].rsplit("-c", 1)[1].removesuffix(".json"))
            row["max_concurrency"] = source_concurrency
            by_concurrency.setdefault(source_concurrency, []).append(row)
        for concurrency, cell_rows in sorted(by_concurrency.items()):
            throughput_winner = max(cell_rows, key=lambda item: item["output_throughput_per_gpu"])
            node_winner = max(cell_rows, key=lambda item: item["projected_eight_gpu_node_output_throughput"])
            latency_winner = min(cell_rows, key=lambda item: item["p50_tpot_ms"])
            comparisons.append(
                {
                    "input_tokens": input_tokens,
                    "latency_winner": latency_winner["topology"],
                    "max_concurrency": concurrency,
                    "node_throughput_winner": node_winner["topology"],
                    "output_tokens": output_tokens,
                    "per_gpu_throughput_winner": throughput_winner["topology"],
                }
            )
    return {"comparisons": comparisons, "model_family": model, "rows": rows, "status": "PASS"}


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True, type=Path)
    parser.add_argument("--model-family", required=True, choices=("dense", "moe"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    summary = summarize(load_rows(args.result_dir, args.model_family), args.model_family)
    atomic_json(args.output, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
