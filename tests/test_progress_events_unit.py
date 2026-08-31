import importlib.util
import json
import os
import sys
import threading
from pathlib import Path


MODULE_PATH = Path(__file__).parents[1] / "skyrl" / "train" / "utils" / "progress_events.py"
SPEC = importlib.util.spec_from_file_location("skyrl_progress_events", MODULE_PATH)
assert SPEC and SPEC.loader
progress_events = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = progress_events
SPEC.loader.exec_module(progress_events)
emit_progress_event = progress_events.emit_progress_event
reset_progress_event_writer_for_tests = progress_events.reset_progress_event_writer_for_tests


def test_progress_events_are_opt_in(tmp_path, monkeypatch):
    monkeypatch.delenv("SKYRL_PROGRESS_EVENTS_PATH", raising=False)
    emit_progress_event("ignored", global_step=1)
    assert list(tmp_path.iterdir()) == []


def test_progress_events_append_valid_jsonl(tmp_path, monkeypatch):
    path = tmp_path / "progress" / "events.jsonl"
    monkeypatch.setenv("SKYRL_PROGRESS_EVENTS_PATH", str(path))
    monkeypatch.setenv("SKYRL_PROGRESS_RUN_ID", "test-run")
    reset_progress_event_writer_for_tests()

    def emit(index):
        emit_progress_event(
            "trajectory_finished",
            global_step=2,
            group_uid="task-a",
            repetition_id=index,
            reward=index / 10,
        )

    threads = [threading.Thread(target=emit, args=(index,)) for index in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 16
    assert {record["repetition_id"] for record in records} == set(range(16))
    assert {record["run_id"] for record in records} == {"test-run"}
    assert {record["schema_version"] for record in records} == {1}
    assert {record["pid"] for record in records} == {os.getpid()}
    assert all(record["hostname"] for record in records)


def test_progress_event_path_supports_process_sharding(tmp_path, monkeypatch):
    path_template = tmp_path / "{hostname}" / "events-{pid}.jsonl"
    monkeypatch.setenv("SKYRL_PROGRESS_EVENTS_PATH", str(path_template))
    reset_progress_event_writer_for_tests()

    emit_progress_event("worker_started", global_step=3)

    paths = list(tmp_path.glob("*/events-*.jsonl"))
    assert len(paths) == 1
    record = json.loads(paths[0].read_text(encoding="utf-8"))
    assert paths[0].name == f"events-{os.getpid()}.jsonl"
    assert paths[0].parent.name == record["hostname"]


def test_progress_events_keep_only_bounded_json_scalars(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    monkeypatch.setenv("SKYRL_PROGRESS_EVENTS_PATH", str(path))
    reset_progress_event_writer_for_tests()

    emit_progress_event(
        "safe_event",
        global_step=4,
        optional=None,
        nested={"credential": "must-not-be-written"},
        sequence=["must-not-be-written"],
        non_finite=float("inf"),
        long_value="x" * 5000,
    )

    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["global_step"] == 4
    assert record["optional"] is None
    assert len(record["long_value"]) == 4096
    assert "nested" not in record
    assert "sequence" not in record
    assert "non_finite" not in record
    assert "must-not-be-written" not in path.read_text(encoding="utf-8")


def test_progress_event_write_failures_never_escape(tmp_path, monkeypatch):
    blocking_file = tmp_path / "not-a-directory"
    blocking_file.write_text("block", encoding="utf-8")
    monkeypatch.setenv("SKYRL_PROGRESS_EVENTS_PATH", str(blocking_file / "events.jsonl"))
    reset_progress_event_writer_for_tests()

    emit_progress_event("first")
    emit_progress_event("second")

    assert blocking_file.read_text(encoding="utf-8") == "block"
