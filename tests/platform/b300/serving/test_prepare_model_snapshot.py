import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = ROOT / "platform/b300/serving/prepare_model_snapshot.py"
SPEC = importlib.util.spec_from_file_location("prepare_model_snapshot", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
prepare_model_snapshot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_model_snapshot)


def make_snapshot(root: Path):
    (root / "config.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (root / "model-00001-of-00001.safetensors").write_bytes(b"weights")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"model.weight": "model-00001-of-00001.safetensors"}}),
        encoding="utf-8",
    )


def test_weight_index_accepts_complete_shards(tmp_path):
    make_snapshot(tmp_path)
    prepare_model_snapshot.validate_weight_index(tmp_path)


def test_weight_index_rejects_missing_shard(tmp_path):
    make_snapshot(tmp_path)
    (tmp_path / "model-00001-of-00001.safetensors").unlink()
    with pytest.raises(prepare_model_snapshot.SnapshotError, match="missing shards"):
        prepare_model_snapshot.validate_weight_index(tmp_path)


def test_manifest_round_trip_and_freeze(tmp_path):
    make_snapshot(tmp_path)
    revision = "a" * 40
    manifest = prepare_model_snapshot.create_manifest(
        tmp_path,
        destination=tmp_path,
        repo_id="Qwen/test",
        revision=revision,
        image_ref="example:test",
        image_id="sha256:" + "b" * 64,
    )
    assert manifest["total_size"] > 0
    prepare_model_snapshot.freeze(tmp_path)
    validated = prepare_model_snapshot.validate_manifest(
        tmp_path,
        repo_id="Qwen/test",
        revision=revision,
        image_ref="example:test",
        image_id="sha256:" + "b" * 64,
    )
    assert validated["files"] == manifest["files"]
    assert not (os.stat(tmp_path).st_mode & 0o222)


def test_manifest_rejects_extra_file(tmp_path):
    make_snapshot(tmp_path)
    prepare_model_snapshot.create_manifest(
        tmp_path,
        destination=tmp_path,
        repo_id="Qwen/test",
        revision="a" * 40,
        image_ref="example:test",
        image_id="sha256:" + "b" * 64,
    )
    (tmp_path / "unexpected").write_text("drift", encoding="utf-8")
    with pytest.raises(prepare_model_snapshot.SnapshotError, match="file set differs"):
        prepare_model_snapshot.validate_manifest(
            tmp_path,
            repo_id="Qwen/test",
            revision="a" * 40,
            image_ref="example:test",
            image_id="sha256:" + "b" * 64,
        )
