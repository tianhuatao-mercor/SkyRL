import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = ROOT / "platform/b300/serving/summarize_fullnode_results.py"
SPEC = importlib.util.spec_from_file_location("summarize_fullnode_results", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
summarizer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summarizer)


def result(phase, throughput, completed=4, failed=0):
    return {
        "completed": completed,
        "duration": 2.0,
        "failed": failed,
        "gpu_count": "8",
        "input_tokens": "1024",
        "mean_itl_ms": 2.0,
        "mean_tpot_ms": 3.0,
        "mean_ttft_ms": 4.0,
        "model_family": "dense",
        "num_prompts": 4,
        "output_throughput": throughput,
        "output_tokens": "256",
        "p50_tpot_ms": 2.5 if phase == "tp8" else 3.0,
        "p50_ttft_ms": 3.5 if phase == "r8tp1" else 4.0,
        "p90_tpot_ms": 4.0,
        "p90_ttft_ms": 5.0,
        "p99_tpot_ms": 6.0,
        "p99_ttft_ms": 7.0,
        "request_throughput": 2.0,
        "topology": phase,
    }


def test_summary_compares_measured_node_results(tmp_path):
    (tmp_path / "r8tp1-i1024-o256-c128.json").write_text(
        json.dumps(result("r8tp1", 800)), encoding="utf-8"
    )
    (tmp_path / "tp8-i1024-o256-c128.json").write_text(
        json.dumps(result("tp8", 500)), encoding="utf-8"
    )
    rows = summarizer.load_rows(tmp_path, "dense")
    summary = summarizer.summarize(rows, "dense", {"r8tp1", "tp8"})
    comparison = summary["comparisons"][0]
    assert comparison["node_throughput_winner"] == "r8tp1"
    assert comparison["p50_tpot_winner"] == "tp8"
    assert comparison["p50_ttft_winner"] == "r8tp1"


def test_missing_phase_fails_closed(tmp_path):
    (tmp_path / "r8tp1-i1024-o256-c128.json").write_text(
        json.dumps(result("r8tp1", 800)), encoding="utf-8"
    )
    rows = summarizer.load_rows(tmp_path, "dense")
    with pytest.raises(summarizer.ResultError, match="phase mismatch"):
        summarizer.summarize(rows, "dense", {"r8tp1", "tp8"})


def test_failed_requests_fail_closed(tmp_path):
    (tmp_path / "tp8-i1024-o256-c128.json").write_text(
        json.dumps(result("tp8", 500, completed=3, failed=1)), encoding="utf-8"
    )
    with pytest.raises(summarizer.ResultError, match="incomplete benchmark"):
        summarizer.load_rows(tmp_path, "dense")
