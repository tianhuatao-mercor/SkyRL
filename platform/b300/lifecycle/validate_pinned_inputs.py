#!/usr/bin/env python3
"""Validate the frozen B300 lifecycle model and dataset artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--dataset-output", type=Path, required=True)
    args = parser.parse_args()

    model_manifest_path = args.model_dir / "MODEL_MANIFEST.json"
    model_manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
    if model_manifest["resolved_revision"] != args.model_revision:
        raise RuntimeError("model revision mismatch")
    for record in model_manifest["files"]:
        path = args.model_dir / record["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.stat().st_size != record["size"] or digest != record["sha256"]:
            raise RuntimeError(f"model artifact mismatch: {path}")
    _atomic_json(
        args.model_output,
        {
            "files": len(model_manifest["files"]),
            "resolved_revision": model_manifest["resolved_revision"],
            "status": "PASS",
        },
    )

    expected_files = {
        "train.parquet": {
            "rows": 4,
            "sha256": "dda98b10be9a17394459b72ba6d8f096bf8e67bdefb18e25f97da3b9a1e9ceff",
            "size": 6049,
        },
        "validation.parquet": {
            "rows": 2,
            "sha256": "11c361118cb3e605b222555739c2fa1eb06c6d6208b37a3abd4375965b41d982",
            "size": 5653,
        },
    }
    dataset_manifest_path = args.dataset_dir / "manifest.json"
    dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
    if dataset_manifest.get("artifact") != "skyrl-multiply-lifecycle-v1" or dataset_manifest.get("schema_version") != 1:
        raise RuntimeError("dataset artifact/schema mismatch")
    if dataset_manifest.get("files") != expected_files:
        raise RuntimeError("dataset file manifest mismatch")
    content_id = hashlib.sha256(
        json.dumps(expected_files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if dataset_manifest.get("content_id") != args.dataset_id or content_id != args.dataset_id:
        raise RuntimeError("dataset content ID mismatch")
    if args.dataset_dir.name != f"skyrl-multiply-lifecycle-{args.dataset_id[:16]}":
        raise RuntimeError("dataset directory/content ID mismatch")
    if args.dataset_dir.is_symlink() or args.dataset_dir.stat().st_mode & 0o222:
        raise RuntimeError("dataset directory is symlinked or writable")
    if {path.name for path in args.dataset_dir.iterdir()} != {"manifest.json", *expected_files}:
        raise RuntimeError("dataset contains unexpected or missing entries")
    for name, record in {"manifest.json": None, **expected_files}.items():
        path = args.dataset_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
            raise RuntimeError(f"dataset artifact is symlinked, missing, or writable: {path}")
        if record and (
            path.stat().st_size != record["size"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]
        ):
            raise RuntimeError(f"dataset artifact mismatch: {path}")
    _atomic_json(
        args.dataset_output,
        {"content_id": dataset_manifest["content_id"], "files": expected_files, "status": "PASS"},
    )


if __name__ == "__main__":
    main()
