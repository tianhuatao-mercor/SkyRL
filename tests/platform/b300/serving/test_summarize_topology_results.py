import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = ROOT / "platform/b300/serving/summarize_topology_results.py"
SPEC = importlib.util.spec_from_file_location("summarize_topology_results", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
summarizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarizer)


def result(topology, gpu_count, throughput, completed=4, failed=0):
    return {
        "completed": completed,
        "duration": 2.0,
        "failed": failed,
        "gpu_count": str(gpu_count),
        "input_tokens": "1024",
        "mean_itl_ms": 2.0,
        "mean_tpot_ms": 3.0,
        "mean_ttft_ms": 4.0,
        "model_family": "dense",
        "num_prompts": 4,
        "output_throughput": throughput,
        "output_tokens": "256",
        "p50_tpot_ms": 2.5 if topology == "tp2" else 3.0,
        "p50_ttft_ms": 3.5,
        "p90_tpot_ms": 4.0,
        "p90_ttft_ms": 5.0,
        "p99_tpot_ms": 6.0,
        "p99_ttft_ms": 7.0,
        "request_throughput": 2.0,
        "topology": topology,
    }


def test_summary_separates_efficiency_latency_and_node_projection(tmp_path):
    (tmp_path / "tp1-i1024-o256-c1.json").write_text(json.dumps(result("tp1", 1, 100)), encoding="utf-8")
    (tmp_path / "tp2-i1024-o256-c1.json").write_text(json.dumps(result("tp2", 2, 160)), encoding="utf-8")
    rows = summarizer.load_rows(tmp_path, "dense")
    summary = summarizer.summarize(rows, "dense")
    comparison = summary["comparisons"][0]
    assert comparison["per_gpu_throughput_winner"] == "tp1"
    assert comparison["node_throughput_winner"] == "tp1"
    assert comparison["latency_winner"] == "tp2"


def test_failed_requests_fail_closed(tmp_path):
    (tmp_path / "tp1-i1024-o256-c1.json").write_text(
        json.dumps(result("tp1", 1, 100, completed=3, failed=1)), encoding="utf-8"
    )
    with pytest.raises(summarizer.ResultError, match="incomplete benchmark"):
        summarizer.load_rows(tmp_path, "dense")


def test_non_divisor_gpu_count_fails_closed(tmp_path):
    (tmp_path / "tp3-i1024-o256-c1.json").write_text(json.dumps(result("tp3", 3, 100)), encoding="utf-8")
    with pytest.raises(summarizer.ResultError, match="unsupported GPU count"):
        summarizer.load_rows(tmp_path, "dense")
