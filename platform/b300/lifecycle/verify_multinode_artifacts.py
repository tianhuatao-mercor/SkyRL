#!/usr/bin/env python3
"""Verify two-node placement, EFA transport, and shutdown evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from urllib.parse import urlparse


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _require_clean_snapshot(path: Path, run_id: str) -> None:
    record = _json(path)
    if record.get("gpu_processes"):
        raise RuntimeError(f"GPU processes remain in {path}: {record['gpu_processes']}")
    if record.get("run_owned_containers"):
        raise RuntimeError(f"run-owned containers remain in {path}: {record['run_owned_containers']}")
    if record.get("conflicting_processes"):
        raise RuntimeError(f"run processes remain in {path}: {record['conflicting_processes']}")
    if record.get("run_id") != run_id:
        raise RuntimeError(f"snapshot run ID mismatch in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--nccl-log-dir", type=Path, required=True)
    parser.add_argument("--head-ip", required=True)
    parser.add_argument("--worker-ip", required=True)
    parser.add_argument("--head-hostname", required=True)
    parser.add_argument("--worker-hostname", required=True)
    parser.add_argument("--driver-resource", required=True)
    parser.add_argument("--rollout-resource", required=True)
    parser.add_argument("--post-head-snapshot", type=Path, required=True)
    parser.add_argument("--post-worker-snapshot", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    preflight = _json(args.result_dir / "ray-cluster-preflight.json")
    if preflight.get("status") != "PASS" or preflight.get("expected_ips") != [args.head_ip, args.worker_ip]:
        raise RuntimeError(f"invalid Ray preflight record: {preflight}")
    nodes = {node["node_manager_address"]: node for node in preflight["nodes"]}
    if set(nodes) != {args.head_ip, args.worker_ip}:
        raise RuntimeError(f"Ray node set mismatch: {sorted(nodes)}")
    head_resources = nodes[args.head_ip]["resources"]
    worker_resources = nodes[args.worker_ip]["resources"]
    if not (
        head_resources.get("GPU") == 1
        and head_resources.get(args.driver_resource, 0) >= 1
        and head_resources.get(args.rollout_resource, 0) == 0
        and worker_resources.get("GPU") == 2
        and worker_resources.get(args.rollout_resource, 0) >= 1
        and worker_resources.get(args.driver_resource, 0) == 0
    ):
        raise RuntimeError(f"Ray role resource mismatch: head={head_resources}, worker={worker_resources}")

    driver = _json(args.result_dir / "driver-placement.json")
    if driver.get("node_id") != nodes[args.head_ip]["node_id"] or driver.get("node_manager_address") != args.head_ip:
        raise RuntimeError(f"driver was not pinned to the head node: {driver}")

    rollout = _json(args.result_dir / "rollout-placement.json")
    if (
        rollout.get("resource_name") != args.rollout_resource
        or rollout.get("strategy") != "STRICT_PACK"
        or len(rollout.get("bundle_specs", [])) != 2
        or rollout.get("bundle_node_ids") != [nodes[args.worker_ip]["node_id"]] * 2
        or len(set(rollout.get("bundle_gpu_ids", []))) != 2
    ):
        raise RuntimeError(f"rollout placement mismatch: {rollout}")
    for bundle in rollout["bundle_specs"]:
        if bundle != {"CPU": 1, "GPU": 1, args.rollout_resource: 0.001}:
            raise RuntimeError(f"unexpected rollout bundle: {bundle}")

    topology = _json(args.result_dir / "runtime-topology.json")
    if topology.get("policy_node_ids") != [nodes[args.head_ip]["node_id"]]:
        raise RuntimeError(f"policy was not placed on the head node: {topology.get('policy_node_ids')}")
    if len(topology.get("policy_gpu_ids", [])) != 1:
        raise RuntimeError(f"unexpected policy GPU evidence: {topology.get('policy_gpu_ids')}")
    master_endpoints = topology.get("policy_master_endpoints", [])
    if len(master_endpoints) != 1 or master_endpoints[0][0] != args.head_ip:
        raise RuntimeError(f"policy rendezvous did not use the head private IP: {master_endpoints}")
    server_urls = topology.get("inference_server_urls", [])
    if len(server_urls) != 2 or len(set(server_urls)) != 2:
        raise RuntimeError(f"unexpected inference URLs: {server_urls}")
    if {urlparse(url).hostname for url in server_urls} != {args.worker_ip}:
        raise RuntimeError(f"inference servers were not advertised from the rollout worker: {server_urls}")

    nccl_logs = sorted(args.nccl_log_dir.glob("nccl-*.log"))
    if not nccl_logs:
        raise RuntimeError("no NCCL logs were captured")
    logs_by_host: dict[str, str] = {}
    for hostname in (args.head_hostname, args.worker_hostname):
        matching = [path for path in nccl_logs if hostname in path.name]
        if not matching:
            raise RuntimeError(f"no NCCL log found for {hostname}")
        logs_by_host[hostname] = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in matching)
    required_transport = (
        "NET/OFI Initializing aws-ofi-nccl 1.20.0",
        "Selected provider is efa, fabric is efa-direct (found 16 nics)",
        "Using network Libfabric",
        "GPU Direct RDMA Enabled",
        "nranks 3",
        "via NET/Libfabric",
        "/GDRDMA",
    )
    for hostname, text in logs_by_host.items():
        missing = [marker for marker in required_transport if marker not in text]
        if missing:
            raise RuntimeError(f"missing EFA/NCCL markers for {hostname}: {missing}")
        forbidden = [marker for marker in ("Using network Socket", "via NET/Socket", "NCCL error") if marker in text]
        if forbidden:
            raise RuntimeError(f"forbidden NCCL fallback/error markers for {hostname}: {forbidden}")

    _require_clean_snapshot(args.post_head_snapshot, args.run_id)
    _require_clean_snapshot(args.post_worker_snapshot, args.run_id)

    result = {
        "checks": {
            "driver_node": args.head_ip,
            "efa_hosts": [args.head_hostname, args.worker_hostname],
            "inference_node": args.worker_ip,
            "policy_node": args.head_ip,
            "ray_resources": {"head": head_resources, "worker": worker_resources},
            "rollout_bundle_gpu_ids": rollout["bundle_gpu_ids"],
            "server_urls": server_urls,
            "shutdown_nodes": [args.head_ip, args.worker_ip],
        },
        "status": "PASS",
    }
    _atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
