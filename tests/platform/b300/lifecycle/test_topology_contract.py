import copy
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[4]
MODULE_PATH = ROOT / "platform/b300/lifecycle/topology_contract.py"
TOPOLOGY_PATH = ROOT / "platform/b300/lifecycle/topologies/aws-b300-20260826-two-node.json"
SPEC = importlib.util.spec_from_file_location("b300_topology_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
topology_contract = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(topology_contract)


def topology():
    return json.loads(TOPOLOGY_PATH.read_text(encoding="utf-8"))


def test_current_topology_is_valid_and_pinned():
    contract = topology_contract.load_contract(TOPOLOGY_PATH)
    assert contract["contract_id"] == "aws-b300-20260826-two-node-one-policy-two-rollout"
    assert contract["roles"]["head"]["host_gpu_ids"] == [2]
    assert contract["roles"]["rollout"]["host_gpu_ids"] == [0, 1]
    assert len(topology_contract.contract_sha256(TOPOLOGY_PATH)) == 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value["roles"]["rollout"].update(private_ip="8.8.8.8"), "private IPv4"),
        (lambda value: value["roles"]["head"].update(host_gpu_ids=[2, 2]), "duplicates"),
        (lambda value: value["roles"]["head"].update(host_gpu_ids=[8]), "expected_host_gpu_count"),
        (lambda value: value["roles"]["rollout"].update(host_gpu_ids=[0]), "exactly two rollout"),
        (lambda value: value.update(schema_version=True), "schema_version"),
        (
            lambda value: value["ray"].update(rollout_resource=value["ray"]["driver_resource"]),
            "must differ",
        ),
        (
            lambda value: value["roles"]["rollout"].update(ssh_alias=value["roles"]["head"]["ssh_alias"]),
            "distinct ssh_alias",
        ),
    ],
)
def test_invalid_topology_fails_closed(mutation, message):
    value = copy.deepcopy(topology())
    mutation(value)
    with pytest.raises(topology_contract.ContractError, match=message):
        topology_contract.validate_contract(value)


def test_unknown_keys_require_a_schema_change():
    value = topology()
    value["roles"]["head"]["undocumented"] = True
    with pytest.raises(topology_contract.ContractError, match="extra=.*undocumented"):
        topology_contract.validate_contract(value)
