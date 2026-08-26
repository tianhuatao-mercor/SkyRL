#!/usr/bin/env python3
"""Build the deterministic, content-addressed SkyRL lifecycle canary dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import datasets
import pyarrow
from datasets import Dataset


SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a helpful assistant that solves multiplication problems. "
        "Show concise reasoning and put the final answer in \\boxed{answer} format."
    ),
}

TRAIN_CASES = ((12, 13), (17, 19), (23, 31), (37, 41))
VALIDATION_CASES = ((14, 16), (29, 33))


def _rows(cases: tuple[tuple[int, int], ...], split: str) -> list[dict]:
    return [
        {
            "data_source": "b300_lifecycle_multiply_v1",
            "prompt": [SYSTEM_PROMPT, {"role": "user", "content": f"{left} * {right}"}],
            "env_class": "multiply",
            "reward_spec": {"method": "rule", "ground_truth": str(left * right)},
            "extra_info": {
                "case_id": f"{split}-{index:02d}",
                "left": left,
                "right": right,
                "split": split,
            },
        }
        for index, (left, right) in enumerate(cases)
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _validate_existing(path: Path, manifest: dict) -> None:
    actual = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if actual != manifest:
        raise RuntimeError(f"existing dataset manifest differs: {path}")
    for name, expected in manifest["files"].items():
        if _sha256(path / name) != expected["sha256"]:
            raise RuntimeError(f"existing dataset hash mismatch: {path / name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/shared/datasets"))
    args = parser.parse_args()

    args.root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".skyrl-lifecycle-", dir=args.root))
    try:
        split_rows = {
            "train": _rows(TRAIN_CASES, "train"),
            "validation": _rows(VALIDATION_CASES, "validation"),
        }
        for split, rows in split_rows.items():
            Dataset.from_list(rows).to_parquet(staging / f"{split}.parquet")

        files = {
            f"{split}.parquet": {
                "rows": len(rows),
                "sha256": _sha256(staging / f"{split}.parquet"),
                "size": (staging / f"{split}.parquet").stat().st_size,
            }
            for split, rows in split_rows.items()
        }
        identity = hashlib.sha256(
            json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        destination = args.root / f"skyrl-multiply-lifecycle-{identity[:16]}"
        manifest = {
            "artifact": "skyrl-multiply-lifecycle-v1",
            "content_id": identity,
            "datasets_version": datasets.__version__,
            "files": files,
            "generator": "platform/b300/lifecycle/build_dataset.py",
            "pyarrow_version": pyarrow.__version__,
            "schema_version": 1,
        }
        _write_json(staging / "manifest.json", manifest)

        if destination.exists():
            _validate_existing(destination, manifest)
            print(destination)
            return

        staging.replace(destination)
        staging = None
        for item in destination.iterdir():
            item.chmod(0o444)
        destination.chmod(0o555)
        _validate_existing(destination, manifest)
        print(destination)
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
