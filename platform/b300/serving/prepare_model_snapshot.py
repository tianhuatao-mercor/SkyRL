#!/usr/bin/env python3
"""Stage and validate one pinned public Hugging Face model snapshot."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import stat
import sys
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


MODEL_ROOT = Path("/shared/models")


class SnapshotError(RuntimeError):
    """A snapshot is unsafe, incomplete, or does not match its manifest."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_files(root: Path, manifest_path: Path | None = None) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise SnapshotError(f"snapshot contains a symlink: {path}")
        if path.is_file() and path != manifest_path:
            relative = path.relative_to(root).as_posix()
            files[relative] = path
    return files


def validate_weight_index(root: Path) -> None:
    required = ("config.json", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise SnapshotError(f"snapshot is missing required files: {missing}")
    index = json.loads((root / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise SnapshotError("model.safetensors.index.json has no weight_map")
    shard_names = set(weight_map.values())
    if not shard_names or any(not isinstance(name, str) or not name.endswith(".safetensors") for name in shard_names):
        raise SnapshotError("weight_map contains an invalid shard name")
    missing_shards = sorted(name for name in shard_names if not (root / name).is_file())
    if missing_shards:
        raise SnapshotError(f"weight_map references missing shards: {missing_shards}")
    actual_shards = {path.name for path in root.glob("*.safetensors")}
    if actual_shards != shard_names:
        raise SnapshotError(
            f"weight shard set differs from index: missing={sorted(shard_names - actual_shards)}, "
            f"extra={sorted(actual_shards - shard_names)}"
        )


def validate_manifest(
    root: Path,
    *,
    repo_id: str,
    revision: str,
    image_ref: str,
    image_id: str,
) -> dict:
    manifest_path = root / "MODEL_MANIFEST.json"
    if not root.is_dir() or not manifest_path.is_file():
        raise SnapshotError(f"snapshot or manifest missing: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": revision,
        "model_path": str(root),
        "image_ref": image_ref,
        "image_id": image_id,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise SnapshotError(f"manifest {key} mismatch: expected {value!r}, got {manifest.get(key)!r}")
    records = manifest.get("files")
    if not isinstance(records, list) or not records:
        raise SnapshotError("manifest files must be a non-empty list")
    recorded: dict[str, dict] = {}
    for record in records:
        relative = record.get("path")
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or ".." in Path(relative).parts
            or relative in recorded
        ):
            raise SnapshotError(f"unsafe or duplicate manifest path: {relative!r}")
        recorded[relative] = record
    actual = snapshot_files(root, manifest_path)
    if set(actual) != set(recorded):
        raise SnapshotError(
            f"manifest file set differs: missing={sorted(set(recorded) - set(actual))}, "
            f"extra={sorted(set(actual) - set(recorded))}"
        )
    total_size = 0
    for relative, path in actual.items():
        record = recorded[relative]
        size = path.stat().st_size
        total_size += size
        if record.get("size") != size:
            raise SnapshotError(f"size mismatch for {relative}")
        if record.get("sha256") != sha256_file(path):
            raise SnapshotError(f"SHA-256 mismatch for {relative}")
        if path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
            raise SnapshotError(f"snapshot file is writable: {relative}")
    if manifest.get("total_size") != total_size:
        raise SnapshotError(f"total size mismatch: expected {manifest.get('total_size')}, got {total_size}")
    if root.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise SnapshotError(f"snapshot root is writable: {root}")
    validate_weight_index(root)
    return manifest


def create_manifest(
    stage: Path,
    *,
    destination: Path,
    repo_id: str,
    revision: str,
    image_ref: str,
    image_id: str,
) -> dict:
    validate_weight_index(stage)
    files = []
    for relative, path in snapshot_files(stage).items():
        files.append({"path": relative, "sha256": sha256_file(path), "size": path.stat().st_size})
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": files,
        "image_id": image_id,
        "image_ref": image_ref,
        "model_path": str(destination),
        "repo_id": repo_id,
        "requested_revision": revision,
        "resolved_revision": revision,
        "schema_version": 1,
        "total_size": sum(record["size"] for record in files),
    }
    temporary = stage / ".MODEL_MANIFEST.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, stage / "MODEL_MANIFEST.json")
    return manifest


def freeze(root: Path) -> None:
    for path in root.rglob("*"):
        os.chmod(path, 0o555 if path.is_dir() else 0o444)
    os.chmod(root, 0o555)


def prepare(args: argparse.Namespace) -> dict:
    destination = args.destination.resolve(strict=False)
    if destination.parent != MODEL_ROOT or destination.name.startswith("."):
        raise SnapshotError(f"destination must be one named directory directly under {MODEL_ROOT}")
    MODEL_ROOT.mkdir(mode=0o755, parents=True, exist_ok=True)
    lock_path = MODEL_ROOT / f".{destination.name}.prepare.lock"
    with lock_path.open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SnapshotError(f"another writer holds {lock_path}") from exc
        if destination.exists():
            manifest = validate_manifest(
                destination,
                repo_id=args.repo_id,
                revision=args.revision,
                image_ref=args.image_ref,
                image_id=args.image_id,
            )
            return {"existing_snapshot": True, "manifest": manifest, "status": "PASS"}

        stage = MODEL_ROOT / f".{destination.name}.staging.{args.run_id}"
        if stage.exists():
            raise SnapshotError(f"refusing existing staging path: {stage}")
        stage.mkdir(mode=0o755)
        try:
            info = HfApi().model_info(repo_id=args.repo_id, revision=args.revision, files_metadata=True)
            if info.private or info.gated:
                raise SnapshotError(f"refusing private or gated repository: {args.repo_id}")
            if info.sha != args.revision:
                raise SnapshotError(f"resolved revision mismatch: expected {args.revision}, got {info.sha}")
            snapshot_download(
                repo_id=args.repo_id,
                revision=args.revision,
                local_dir=stage,
                max_workers=args.max_workers,
            )
            shutil.rmtree(stage / ".cache", ignore_errors=True)
            manifest = create_manifest(
                stage,
                destination=destination,
                repo_id=args.repo_id,
                revision=args.revision,
                image_ref=args.image_ref,
                image_id=args.image_id,
            )
            freeze(stage)
            if destination.exists():
                raise SnapshotError(f"destination appeared during preparation: {destination}")
            stage.rename(destination)
            validate_manifest(
                destination,
                repo_id=args.repo_id,
                revision=args.revision,
                image_ref=args.image_ref,
                image_id=args.image_id,
            )
            return {"existing_snapshot": False, "manifest": manifest, "status": "PASS"}
        except BaseException:
            if stage.exists():
                for path in stage.rglob("*"):
                    try:
                        os.chmod(path, 0o755 if path.is_dir() else 0o644)
                    except OSError:
                        pass
                shutil.rmtree(stage)
            raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--image-ref", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-workers", type=int, default=8)
    args = parser.parse_args()
    if not args.run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for character in args.run_id):
        parser.error("--run-id contains unsupported characters")
    if len(args.revision) != 40 or any(character not in "0123456789abcdef" for character in args.revision):
        parser.error("--revision must be a full lowercase Git revision")
    if args.max_workers < 1 or args.max_workers > 32:
        parser.error("--max-workers must be between 1 and 32")
    try:
        result = prepare(args)
    except (OSError, SnapshotError, ValueError) as exc:
        print(json.dumps({"error": str(exc), "status": "FAIL"}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
