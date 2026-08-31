#!/usr/bin/env python3
"""Read-only live preflight for a validated two-node lifecycle topology."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from topology_contract import ContractError, contract_sha256, load_contract


SSH_OPTIONS = ("-o", "BatchMode=yes", "-o", "ConnectTimeout=20")


class PreflightError(RuntimeError):
    """A live node does not match the declared topology contract."""


def _remote(alias: str, *args: str) -> str:
    command = ["ssh", *SSH_OPTIONS, alias, shlex.join(args)]
    result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}"
        raise PreflightError(f"{alias}: {shlex.join(args)} failed: {detail}")
    return result.stdout.rstrip("\n")


def _remote_shell(alias: str, script: str) -> str:
    return _remote(alias, "bash", "-lc", script)


def _gpu_rows(output: str, alias: str) -> list[dict[str, int]]:
    rows: list[dict[str, int]] = []
    for line in output.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2:
            raise PreflightError(f"{alias}: unexpected nvidia-smi row: {line!r}")
        try:
            rows.append({"index": int(fields[0]), "memory_used_mib": int(fields[1])})
        except ValueError as exc:
            raise PreflightError(f"{alias}: non-integer nvidia-smi row: {line!r}") from exc
    return rows


def inspect_node(role_name: str, role: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    alias = role["ssh_alias"]
    hostname = _remote(alias, "hostname", "-s")
    if hostname != role["hostname"]:
        raise PreflightError(f"{alias}: hostname drift: expected {role['hostname']}, got {hostname}")
    addresses = _remote(alias, "hostname", "-I").split()
    if role["private_ip"] not in addresses:
        raise PreflightError(f"{alias}: private IP {role['private_ip']} not present in {addresses}")

    mountpoint = contract["shared_filesystem"]["mountpoint"]
    mount = _remote(alias, "findmnt", "-T", mountpoint, "-n", "-o", "SOURCE,FSTYPE").split()
    expected_mount = [contract["shared_filesystem"]["source"], contract["shared_filesystem"]["fstype"]]
    if mount != expected_mount:
        raise PreflightError(f"{alias}: shared filesystem drift: expected {expected_mount}, got {mount}")

    gpu_rows = _gpu_rows(
        _remote(alias, "nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"),
        alias,
    )
    if len(gpu_rows) != role["expected_host_gpu_count"]:
        raise PreflightError(
            f"{alias}: expected {role['expected_host_gpu_count']} GPUs, found {len(gpu_rows)}"
        )
    gpu_by_index = {row["index"]: row for row in gpu_rows}
    selected = [gpu_by_index[index] for index in role["host_gpu_ids"]]
    busy = [row for row in selected if row["memory_used_mib"] != 0]
    if busy:
        raise PreflightError(f"{alias}: selected GPUs are not idle: {busy}")
    compute_processes = _remote(
        alias,
        "nvidia-smi",
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader,nounits",
    ).splitlines()
    compute_processes = [line for line in compute_processes if line.strip()]
    if compute_processes:
        raise PreflightError(f"{alias}: GPU compute processes exist: {compute_processes}")

    efa_devices = _remote_shell(
        alias,
        "for path in /sys/class/infiniband/*; do "
        "[ -e \"$path\" ] || continue; "
        "[ \"$(cat \"$path/device/vendor\" 2>/dev/null)\" = 0x1d0f ] || continue; "
        "basename \"$path\"; done | LC_ALL=C sort",
    ).splitlines()
    if len(efa_devices) < role["minimum_efa_devices"]:
        raise PreflightError(
            f"{alias}: expected at least {role['minimum_efa_devices']} EFA device nodes, found {len(efa_devices)}"
        )

    owner = contract["qualification"]["owner_label"]
    owned_containers = _remote(
        alias,
        "docker",
        "ps",
        "-aq",
        "--filter",
        f"label=io.mercor.qualification.owner={owner}",
    ).splitlines()
    owned_containers = [line for line in owned_containers if line.strip()]
    if owned_containers:
        raise PreflightError(f"{alias}: lifecycle-owned containers already exist: {owned_containers}")
    conflicting_processes = _remote_shell(
        alias,
        "ps -eo comm=,args= | awk '"
        "$1 == \"raylet\" || $1 == \"gcs_server\" || "
        "(($1 == \"python\" || $1 == \"python3\") && "
        "($0 ~ /ray::/ || $0 ~ /vllm\\.entrypoints/ || $0 ~ /qualification_entrypoint\\.py/)) {print}'",
    ).splitlines()
    conflicting_processes = [line for line in conflicting_processes if line.strip()]
    if conflicting_processes:
        raise PreflightError(f"{alias}: conflicting Ray/vLLM/SkyRL processes exist")

    runtime = contract["runtime"]
    image_id = _remote(alias, "docker", "image", "inspect", runtime["image_ref"], "--format", "{{.Id}}")
    if image_id != runtime["image_id"]:
        raise PreflightError(f"{alias}: image ID drift: expected {runtime['image_id']}, got {image_id}")
    source_revision = _remote(
        alias,
        "docker",
        "image",
        "inspect",
        runtime["image_id"],
        "--format",
        '{{index .Config.Labels "org.opencontainers.image.revision"}}',
    )
    if source_revision != runtime["source_revision"]:
        raise PreflightError(f"{alias}: image source revision drift")
    lock_sha256 = _remote(
        alias,
        "docker",
        "image",
        "inspect",
        runtime["image_id"],
        "--format",
        '{{index .Config.Labels "io.mercor.skyrl.lock-sha256"}}',
    )
    if lock_sha256 != runtime["lock_sha256"]:
        raise PreflightError(f"{alias}: image dependency-lock drift")

    return {
        "addresses": addresses,
        "compute_processes": compute_processes,
        "efa_device_count": len(efa_devices),
        "efa_devices": efa_devices,
        "gpu_inventory": gpu_rows,
        "hostname": hostname,
        "image_id": image_id,
        "lock_sha256": lock_sha256,
        "owned_containers": owned_containers,
        "private_ip": role["private_ip"],
        "role": role_name,
        "selected_gpu_ids": role["host_gpu_ids"],
        "shared_filesystem": {"fstype": mount[1], "mountpoint": mountpoint, "source": mount[0]},
        "source_revision": source_revision,
        "ssh_alias": alias,
        "timestamp": _remote(alias, "date", "-u", "+%Y-%m-%dT%H:%M:%SZ"),
    }


def run_preflight(topology_path: Path) -> dict[str, Any]:
    contract = load_contract(topology_path)
    nodes = {
        role_name: inspect_node(role_name, contract["roles"][role_name], contract)
        for role_name in ("head", "rollout")
    }
    return {
        "checked_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "contract_id": contract["contract_id"],
        "nodes": nodes,
        "schema_version": contract["schema_version"],
        "status": "PASS",
        "topology_sha256": contract_sha256(topology_path),
    }


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _git(worktree: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(worktree), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def write_evidence(output_dir: Path, topology_path: Path, result: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    recipe_dir = output_dir / "recipe"
    recipe_dir.mkdir()
    script_dir = Path(__file__).resolve().parent
    shutil.copy2(topology_path, output_dir / "topology-source.json")
    shutil.copy2(script_dir / "topology_contract.py", recipe_dir / "topology_contract.py")
    shutil.copy2(Path(__file__), recipe_dir / Path(__file__).name)
    _write_text(output_dir / "preflight-result.json", json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_text(output_dir / "topology-source.sha256", f"{result['topology_sha256']}  topology-source.json\n")

    worktree = Path(_git(script_dir, "rev-parse", "--show-toplevel").strip())
    _write_text(output_dir / "worktree-revision.txt", _git(worktree, "rev-parse", "HEAD"))
    _write_text(output_dir / "worktree-status.txt", _git(worktree, "status", "--short"))
    _write_text(output_dir / "worktree-remotes.txt", _git(worktree, "remote", "-v"))

    evidence_lines = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file() and item.name != "EVIDENCE.sha256"):
        relative = path.relative_to(output_dir)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        evidence_lines.append(f"{digest}  {relative}\n")
    _write_text(output_dir / "EVIDENCE.sha256", "".join(evidence_lines))
    for path in sorted(output_dir.rglob("*"), reverse=True):
        mode = 0o555 if path.is_dir() else 0o444
        os.chmod(path, mode)
    os.chmod(output_dir, 0o555)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        result = run_preflight(args.topology)
        if args.output_dir is not None:
            write_evidence(args.output_dir, args.topology, result)
    except (ContractError, PreflightError, OSError, subprocess.SubprocessError) as exc:
        print(json.dumps({"error": str(exc), "status": "FAIL"}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
