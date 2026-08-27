#!/usr/bin/env python3
"""Validate and summarize common-load, full-node vLLM benchmark results."""

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
                f"incomplete benchmark in {path}: completed={completed}, "
                f"failed={failed}, prompts={prompts}"
            )
        phase = str(source["topology"])
        concurrency = int(path.stem.rsplit("-c", 1)[1])
        row = {
            "completed": int(completed),
            "duration_s": float(source["duration"]),
            "input_tokens": int(source["input_tokens"]),
            "max_concurrency": concurrency,
            "mean_itl_ms": float(source["mean_itl_ms"]),
            "mean_tpot_ms": float(source["mean_tpot_ms"]),
            "mean_ttft_ms": float(source["mean_ttft_ms"]),
            "node_output_throughput": float(source["output_throughput"]),
            "output_tokens": int(source["output_tokens"]),
            "p50_tpot_ms": float(source["p50_tpot_ms"]),
            "p50_ttft_ms": float(source["p50_ttft_ms"]),
            "p90_tpot_ms": float(source["p90_tpot_ms"]),
            "p90_ttft_ms": float(source["p90_ttft_ms"]),
            "p99_tpot_ms": float(source["p99_tpot_ms"]),
            "p99_ttft_ms": float(source["p99_ttft_ms"]),
            "phase": phase,
            "request_throughput": float(source["request_throughput"]),
            "result_file": path.name,
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


def summarize(rows: list[dict], model: str, expected_phases: set[str]) -> dict:
    by_workload: dict[tuple[int, int, int], list[dict]] = {}
    for row in rows:
        key = (row["input_tokens"], row["output_tokens"], row["max_concurrency"])
        by_workload.setdefault(key, []).append(row)

    comparisons = []
    for (input_tokens, output_tokens, concurrency), cell_rows in sorted(by_workload.items()):
        phases = {row["phase"] for row in cell_rows}
        if phases != expected_phases:
            raise ResultError(
                f"phase mismatch for i{input_tokens}-o{output_tokens}-c{concurrency}: "
                f"expected={sorted(expected_phases)}, actual={sorted(phases)}"
            )
        comparisons.append(
            {
                "input_tokens": input_tokens,
                "max_concurrency": concurrency,
                "node_throughput_winner": max(
                    cell_rows, key=lambda item: item["node_output_throughput"]
                )["phase"],
                "output_tokens": output_tokens,
                "p50_tpot_winner": min(cell_rows, key=lambda item: item["p50_tpot_ms"])["phase"],
                "p50_ttft_winner": min(cell_rows, key=lambda item: item["p50_ttft_ms"])["phase"],
            }
        )
    return {
        "comparisons": comparisons,
        "expected_phases": sorted(expected_phases),
        "model_family": model,
        "rows": rows,
        "status": "PASS",
    }


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
    parser.add_argument("--expected-phases", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    expected_phases = {phase for phase in args.expected_phases.split(",") if phase}
    if not expected_phases:
        raise ResultError("at least one expected phase is required")
    summary = summarize(load_rows(args.result_dir, args.model_family), args.model_family, expected_phases)
    atomic_json(args.output, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
