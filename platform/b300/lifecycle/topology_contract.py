#!/usr/bin/env python3
"""Validate the portable inventory and placement contract for the B300 gate."""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
ROLE_NAMES = ("head", "rollout")
HEX_40 = re.compile(r"[0-9a-f]{40}")
HEX_64 = re.compile(r"[0-9a-f]{64}")
SAFE_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
RAY_RESOURCE = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,63}")


class ContractError(ValueError):
    """The topology contract is incomplete or internally inconsistent."""


def _object(value: Any, where: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError(f"{where} must be an object")
    actual = set(value)
    if actual != keys:
        missing = sorted(keys - actual)
        extra = sorted(actual - keys)
        raise ContractError(f"{where} keys differ: missing={missing}, extra={extra}")
    return value


def _string(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{where} must be a non-empty string")
    return value


def _integer(value: Any, where: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ContractError(f"{where} must be an integer in [{minimum}, {maximum}]")
    return value


def _validate_runtime(value: Any) -> None:
    runtime = _object(
        value,
        "runtime",
        {"image_id", "image_ref", "lock_sha256", "source_revision"},
    )
    _string(runtime["image_ref"], "runtime.image_ref")
    image_id = _string(runtime["image_id"], "runtime.image_id")
    if not image_id.startswith("sha256:") or not HEX_64.fullmatch(image_id.removeprefix("sha256:")):
        raise ContractError("runtime.image_id must be a full sha256 image ID")
    if not HEX_40.fullmatch(_string(runtime["source_revision"], "runtime.source_revision")):
        raise ContractError("runtime.source_revision must be a 40-character lowercase Git revision")
    if not HEX_64.fullmatch(_string(runtime["lock_sha256"], "runtime.lock_sha256")):
        raise ContractError("runtime.lock_sha256 must be a lowercase SHA-256 digest")


def _validate_filesystem(value: Any) -> None:
    filesystem = _object(value, "shared_filesystem", {"fstype", "mountpoint", "source"})
    mountpoint = Path(_string(filesystem["mountpoint"], "shared_filesystem.mountpoint"))
    if not mountpoint.is_absolute():
        raise ContractError("shared_filesystem.mountpoint must be absolute")
    _string(filesystem["source"], "shared_filesystem.source")
    _string(filesystem["fstype"], "shared_filesystem.fstype")


def _validate_role(name: str, value: Any) -> None:
    role = _object(
        value,
        f"roles.{name}",
        {
            "expected_host_gpu_count",
            "host_gpu_ids",
            "hostname",
            "minimum_efa_devices",
            "private_ip",
            "ray_cpu_slots",
            "ssh_alias",
        },
    )
    _string(role["ssh_alias"], f"roles.{name}.ssh_alias")
    _string(role["hostname"], f"roles.{name}.hostname")
    try:
        address = ipaddress.ip_address(_string(role["private_ip"], f"roles.{name}.private_ip"))
    except ValueError as exc:
        raise ContractError(f"roles.{name}.private_ip is invalid: {exc}") from exc
    if address.version != 4 or not address.is_private:
        raise ContractError(f"roles.{name}.private_ip must be a private IPv4 address")

    gpu_count = _integer(
        role["expected_host_gpu_count"],
        f"roles.{name}.expected_host_gpu_count",
        minimum=1,
        maximum=256,
    )
    gpu_ids = role["host_gpu_ids"]
    if not isinstance(gpu_ids, list) or not gpu_ids:
        raise ContractError(f"roles.{name}.host_gpu_ids must be a non-empty list")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in gpu_ids):
        raise ContractError(f"roles.{name}.host_gpu_ids must contain only integers")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ContractError(f"roles.{name}.host_gpu_ids contains duplicates")
    if min(gpu_ids) < 0 or max(gpu_ids) >= gpu_count:
        raise ContractError(f"roles.{name}.host_gpu_ids must fit expected_host_gpu_count")
    _integer(role["minimum_efa_devices"], f"roles.{name}.minimum_efa_devices", minimum=1, maximum=256)
    _integer(role["ray_cpu_slots"], f"roles.{name}.ray_cpu_slots", minimum=1, maximum=4096)


def validate_contract(contract: Any) -> dict[str, Any]:
    root = _object(
        contract,
        "contract",
        {
            "contract_id",
            "qualification",
            "ray",
            "roles",
            "runtime",
            "schema_version",
            "shared_filesystem",
        },
    )
    if isinstance(root["schema_version"], bool) or root["schema_version"] != SCHEMA_VERSION:
        raise ContractError(f"schema_version must be exactly {SCHEMA_VERSION}")
    if not SAFE_ID.fullmatch(_string(root["contract_id"], "contract_id")):
        raise ContractError("contract_id contains unsupported characters")

    _validate_runtime(root["runtime"])
    _validate_filesystem(root["shared_filesystem"])

    qualification = _object(root["qualification"], "qualification", {"owner_label"})
    if not SAFE_ID.fullmatch(_string(qualification["owner_label"], "qualification.owner_label")):
        raise ContractError("qualification.owner_label contains unsupported characters")

    ray = _object(
        root["ray"],
        "ray",
        {"driver_resource", "port", "rollout_resource"},
    )
    _integer(ray["port"], "ray.port", minimum=1024, maximum=65535)
    for key in ("driver_resource", "rollout_resource"):
        if not RAY_RESOURCE.fullmatch(_string(ray[key], f"ray.{key}")):
            raise ContractError(f"ray.{key} is not a valid custom Ray resource name")
    if ray["driver_resource"] == ray["rollout_resource"]:
        raise ContractError("driver and rollout Ray resources must differ")

    roles = _object(root["roles"], "roles", set(ROLE_NAMES))
    for name in ROLE_NAMES:
        _validate_role(name, roles[name])
    head = roles["head"]
    rollout = roles["rollout"]
    if len(head["host_gpu_ids"]) != 1:
        raise ContractError("the bounded lifecycle requires exactly one head/policy GPU")
    if len(rollout["host_gpu_ids"]) != 2:
        raise ContractError("the bounded lifecycle requires exactly two rollout GPUs")
    for key in ("ssh_alias", "private_ip", "hostname"):
        if head[key] == rollout[key]:
            raise ContractError(f"head and rollout roles must have distinct {key} values")

    return root


def load_contract(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise ContractError(f"cannot read topology contract {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ContractError(f"invalid JSON in topology contract {path}: {exc}") from exc
    return validate_contract(payload)


def contract_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--sha256-only", action="store_true")
    args = parser.parse_args()
    contract = load_contract(args.topology)
    if args.sha256_only:
        print(contract_sha256(args.topology))
    else:
        print(json.dumps(contract, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
