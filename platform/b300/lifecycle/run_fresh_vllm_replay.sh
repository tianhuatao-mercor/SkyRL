#!/usr/bin/env bash
set -Eeuo pipefail

readonly IMAGE_REF="skyrl-megatron-b300-cu128-canary:7c528991c4f9-r1"
readonly IMAGE_ID="sha256:e98a7978ad815edbd55d460f0f45ec059dcc2df4584e5c8da9c6183b99b2940c"
readonly SOURCE_RUN="20260826T233354Z-skyrl-lifecycle-nccl-dense-r1"
readonly SOURCE_AUDIT="20260826T234504Z-skyrl-lifecycle-audit-r1"
readonly SOURCE_QUAL="/shared/environments/b300/qualifications/$SOURCE_RUN"
readonly SOURCE_CHECKPOINT="/shared/checkpoints/qualifications/$SOURCE_RUN"
readonly SOURCE_AUDIT_DIR="/shared/environments/b300/qualifications/$SOURCE_AUDIT"
readonly EXPORT_PATH="$SOURCE_CHECKPOINT/exports/global_step_1/policy"
readonly EXPECTED_TRAJECTORY="$SOURCE_QUAL/results/trajectory-eval-step-1.json"
readonly QUAL_ROOT="/shared/environments/b300/qualifications"
readonly RECIPE_ROOT="/shared/environments/b300/recipes"
readonly WORKTREE="/shared/code/SkyRL-b300-lifecycle"
readonly OWNER_LABEL="skyrl-fresh-vllm-replay"

die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

execute=false
gpu=0
while (($#)); do
  case "$1" in
    --execute) execute=true; shift ;;
    --gpu) gpu=${2:?}; shift 2 ;;
    -h|--help)
      printf 'Usage: %s --execute [--gpu 0]\n' "$0"
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$execute" == true ]] || die "refusing GPU launch without --execute"
[[ "$gpu" =~ ^[0-7]$ ]] || die "--gpu must be one B300 index from 0 through 7"
[[ $(hostname -s) == ip-172-31-67-119 ]] || die "fresh replay is pinned to aws-b300-node1"
[[ -d "$EXPORT_PATH" && -f "$EXPECTED_TRAJECTORY" ]] || die "pinned source artifacts are missing"
[[ -z $(git -C "$WORKTREE" status --short) ]] || die "lifecycle worktree must be clean"

actual_image_id=$(docker image inspect "$IMAGE_REF" --format '{{.Id}}')
[[ "$actual_image_id" == "$IMAGE_ID" ]] || die "image ID drift: $actual_image_id"

run_id="$(date -u +%Y%m%dT%H%M%SZ)-skyrl-fresh-vllm-replay-r1"
qual_dir="$QUAL_ROOT/$run_id"
scratch_token=$(printf '%s' "$run_id" | sha256sum | cut -c1-12)
scratch_dir="/opt/dlami/nvme/cache/tmp/sr/$scratch_token"
container_name="b300-replay-$run_id"
payload="$WORKTREE/platform/b300/lifecycle/fresh_vllm_replay.py"
payload_hash=$(sha256sum "$payload" | awk '{print $1}')
payload_dir="$RECIPE_ROOT/skyrl-fresh-vllm-replay-$payload_hash"
container_id=""
finalized=false

for path in "$qual_dir" "$scratch_dir"; do
  [[ ! -e "$path" ]] || die "refusing to reuse path: $path"
done
[[ -z $(docker ps -aq --filter "name=^/${container_name}$") ]] || die "container name already exists"
mkdir -p "$qual_dir" "$scratch_dir"/{h,t,r,xdg,triton,vllm}

snapshot() {
  local prefix=$1
  date -u +%Y-%m-%dT%H:%M:%S.%NZ >"$qual_dir/$prefix-timestamp.txt"
  nvidia-smi -q -x >"$qual_dir/$prefix-nvidia-smi.xml"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader >"$qual_dir/$prefix-gpu-processes.csv" || true
  docker ps -a --no-trunc >"$qual_dir/$prefix-containers.txt"
  ps -eo user,pid,ppid,lstart,args --sort=pid >"$qual_dir/$prefix-processes.txt"
  findmnt -T /shared -o TARGET,SOURCE,FSTYPE,OPTIONS >"$qual_dir/$prefix-fsx-mount.txt"
  ss -lntup >"$qual_dir/$prefix-listeners.txt" || true
}

verify_container_identity() {
  [[ $(docker inspect "$container_id" --format '{{index .Config.Labels "io.mercor.qualification.run-id"}}') == "$run_id" ]]
  [[ $(docker inspect "$container_id" --format '{{index .Config.Labels "io.mercor.qualification.owner"}}') == "$OWNER_LABEL" ]]
  [[ $(docker inspect "$container_id" --format '{{.Name}}') == "/$container_name" ]]
}

cleanup_owned_container() {
  [[ -n "$container_id" ]] || return 0
  if docker inspect "$container_id" >/dev/null 2>&1; then
    verify_container_identity || return 1
    if [[ $(docker inspect "$container_id" --format '{{.State.Running}}') == true ]]; then
      docker stop --time 60 "$container_id" >"$qual_dir/container-stop.txt"
    fi
    docker inspect "$container_id" >"$qual_dir/container-final-inspect.json"
    docker logs --timestamps "$container_id" >"$qual_dir/container-final.log" 2>&1 || true
    docker rm "$container_id" >"$qual_dir/container-remove.txt"
  fi
  [[ -z $(docker ps -aq --filter "id=$container_id") ]] || return 1
  container_id=""
}

finalize_evidence() {
  (
    cd "$qual_dir"
    find . -type f ! -name EVIDENCE.sha256 -print0 | sort -z | xargs -0 -r sha256sum
  ) >"$qual_dir/EVIDENCE.sha256"
  chmod -R a-w "$qual_dir"
  finalized=true
}

on_exit() {
  local rc=$?
  if [[ "$finalized" != true ]]; then
    cleanup_owned_container || true
    snapshot post-failure || true
    printf '{"exit_code":%d,"status":"FAIL"}\n' "$rc" >"$qual_dir/launcher-result.json" 2>/dev/null || true
    finalize_evidence 2>/dev/null || true
  fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

snapshot pre

used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
[[ "$used" == 0 ]] || die "GPU $gpu is not idle: ${used} MiB"
gpu_uuid=$(nvidia-smi --id="$gpu" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F, -v uuid="$gpu_uuid" '$1 == uuid {found=1} END {exit !found}'; then
  die "GPU $gpu has an active compute process"
fi
[[ -z $(docker ps -aq --filter "label=io.mercor.qualification.owner=$OWNER_LABEL") ]] || die "another replay container exists"
ps -eo comm=,args= | awk '
  $1 == "raylet" || $1 == "gcs_server" ||
  (($1 == "python" || $1 == "python3") &&
  ($0 ~ /ray::/ || $0 ~ /vllm\.entrypoints/ || $0 ~ /fresh_vllm_replay\.py/)) {print}
' >"$qual_dir/pre-conflicting-processes.txt"
[[ ! -s "$qual_dir/pre-conflicting-processes.txt" ]] || die "conflicting Ray/vLLM/replay processes exist"

(
  cd "$SOURCE_QUAL"
  sha256sum -c EVIDENCE.sha256
) >"$qual_dir/source-evidence-verification.stdout"
(
  cd "$SOURCE_CHECKPOINT"
  sha256sum -c "$SOURCE_QUAL/checkpoint-files.sha256"
) >"$qual_dir/source-checkpoint-verification.stdout"

python3 - "$SOURCE_AUDIT_DIR/artifact-verification.json" "$EXPECTED_TRAJECTORY" "$EXPORT_PATH" "$qual_dir/preflight.json" <<'PY'
import hashlib, json, os, stat, sys
from pathlib import Path
audit_path, trajectory_path, export_path, output_path = map(Path, sys.argv[1:])
audit = json.loads(audit_path.read_text())
if audit.get("status") != "PASS": raise SystemExit("source lifecycle audit is not PASS")
if audit["checks"]["checkpoint"]["state_layout"]["model_layout"] != "megatron_gpt_split":
    raise SystemExit("source checkpoint layout drift")
if audit["checks"]["parameter_delta"]["after_sha256"] != "67cbd4a77741f7c4889e5da65c55e83e740bf257bab08e8e9a9438bfdbe79fba":
    raise SystemExit("source trained parameter digest drift")
if hashlib.sha256(trajectory_path.read_bytes()).hexdigest() != "08a693671e7b25bc7fd342a745d799d17fbb2c2de7aa4c31f4daf5e7c3fbe17a":
    raise SystemExit("source post-update trajectory drift")
expected_files = {
    "chat_template.jinja", "config.json", "generation_config.json", "model.safetensors",
    "tokenizer.json", "tokenizer_config.json",
}
actual_files = {path.name for path in export_path.iterdir() if path.is_file()}
if actual_files != expected_files: raise SystemExit(f"trained export file set drift: {actual_files}")
for path in export_path.iterdir():
    if path.is_symlink() or path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
        raise SystemExit(f"trained export is symlinked or writable: {path}")
record = {
    "artifact_audit": str(audit_path),
    "export_path": str(export_path),
    "files": sorted(expected_files),
    "parameter_digest": audit["checks"]["parameter_delta"]["after_sha256"],
    "source_trajectory": str(trajectory_path),
    "status": "PASS",
}
temporary = output_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
os.replace(temporary, output_path)
PY

if [[ -e "$payload_dir" ]]; then
  [[ $(sha256sum "$payload_dir/fresh_vllm_replay.py" | awk '{print $1}') == "$payload_hash" ]] || die "payload cache drift"
else
  mkdir "$payload_dir"
  install -m 0444 "$payload" "$payload_dir/fresh_vllm_replay.py"
  chmod 0555 "$payload_dir"
fi

install -d "$qual_dir/recipe"
install -m 0444 "$WORKTREE/platform/b300/lifecycle/"{fresh_vllm_replay.py,run_fresh_vllm_replay.sh} "$qual_dir/recipe/"
git -C "$WORKTREE" rev-parse HEAD >"$qual_dir/worktree-revision.txt"
git -C "$WORKTREE" status --short >"$qual_dir/worktree-status.txt"
git -C "$WORKTREE" remote -v >"$qual_dir/worktree-remotes.txt"
docker image inspect "$IMAGE_ID" >"$qual_dir/image-inspect.json"
cp "$SOURCE_QUAL/EVIDENCE.sha256" "$qual_dir/source-EVIDENCE.sha256"
cp "$SOURCE_QUAL/checkpoint-files.sha256" "$qual_dir/source-checkpoint-files.sha256"
printf 'run_id=%s\nsource_run=%s\nsource_audit=%s\nimage_id=%s\ngpu=%s\nexport_path=%s\nscratch_dir=%s\n' \
  "$run_id" "$SOURCE_RUN" "$SOURCE_AUDIT" "$IMAGE_ID" "$gpu" "$EXPORT_PATH" "$scratch_dir" >"$qual_dir/command.txt"

docker_args=(
  create
  --name "$container_name"
  --label "io.mercor.qualification.owner=$OWNER_LABEL"
  --label "io.mercor.qualification.run-id=$run_id"
  --gpus "device=$gpu"
  --network host
  --ipc host
  --user "$(id -u):$(id -g)"
  --ulimit memlock=-1:-1
  --ulimit stack=67108864:67108864
  --mount type=bind,src=/shared,dst=/shared,readonly
  --mount "type=bind,src=$qual_dir,dst=$qual_dir"
  --mount "type=bind,src=$scratch_dir,dst=/c"
  --mount "type=bind,src=$payload_dir/fresh_vllm_replay.py,dst=/opt/qualification/fresh_vllm_replay.py,readonly"
  --env NVIDIA_VISIBLE_DEVICES=0
  --env CUDA_VISIBLE_DEVICES=0
  --env HF_HUB_OFFLINE=1
  --env TRANSFORMERS_OFFLINE=1
  --env HF_DATASETS_OFFLINE=1
  --env HF_HUB_DISABLE_TELEMETRY=1
  --env VLLM_NO_USAGE_STATS=1
  --env VLLM_USE_FLASHINFER_SAMPLER=0
  --env VLLM_ENABLE_V1_MULTIPROCESSING=0
  --env VLLM_ALLOW_INSECURE_SERIALIZATION=1
  --env VLLM_ALLOW_RUNTIME_LORA_UPDATING=1
  --env RAY_USAGE_STATS_ENABLED=0
  --env DO_NOT_TRACK=1
  --env PYTHONDONTWRITEBYTECODE=1
  --env PYTHONNOUSERSITE=1
  --env NO_PROXY=127.0.0.1,localhost
  --env no_proxy=127.0.0.1,localhost
  --env HOME=/c/h
  --env TMPDIR=/c/t
  --env RAY_TMPDIR=/c/r
  --env XDG_CACHE_HOME=/c/xdg
  --env TRITON_CACHE_DIR=/c/triton
  --env VLLM_CACHE_ROOT=/c/vllm
  "$IMAGE_ID"
  python /opt/qualification/fresh_vllm_replay.py
  --run-id "$run_id"
  --export-path "$EXPORT_PATH"
  --expected-trajectory "$EXPECTED_TRAJECTORY"
  --result-path "$qual_dir/replay-result.json"
  --server-log-path "$qual_dir/server.log"
  --startup-timeout-seconds 600
)
printf '%q ' docker "${docker_args[@]}" >"$qual_dir/docker-create-command.txt"
printf '\n' >>"$qual_dir/docker-create-command.txt"
container_id=$(docker "${docker_args[@]}")
verify_container_identity || die "created container identity mismatch"
docker inspect "$container_id" >"$qual_dir/container-prestart-inspect.json"

set +e
timeout --foreground --signal=TERM --kill-after=60s 900s docker start -a "$container_id" 2>&1 | tee "$qual_dir/attached.log"
start_rc=${PIPESTATUS[0]}
set -e
container_exit_code=$(docker inspect "$container_id" --format '{{.State.ExitCode}}')
docker inspect "$container_id" >"$qual_dir/container-postexit-inspect.json"
cleanup_owned_container || die "owned container cleanup failed"
snapshot post

[[ "$start_rc" -eq 0 ]] || die "bounded replay start failed or timed out: rc=$start_rc"
[[ "$container_exit_code" -eq 0 ]] || die "replay container exit code was $container_exit_code"
for _ in $(seq 1 30); do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [[ "$used" == 0 ]] && break
  sleep 2
done
[[ "$used" == 0 ]] || die "GPU $gpu still has ${used} MiB allocated"
[[ -z $(docker ps -aq --filter "label=io.mercor.qualification.run-id=$run_id") ]] || die "run container remains"
ps -eo comm=,args= | awk '
  $1 == "raylet" || $1 == "gcs_server" ||
  (($1 == "python" || $1 == "python3") &&
  ($0 ~ /ray::/ || $0 ~ /vllm\.entrypoints/ || $0 ~ /fresh_vllm_replay\.py/)) {print}
' >"$qual_dir/post-conflicting-processes.txt"
[[ ! -s "$qual_dir/post-conflicting-processes.txt" ]] || die "replay left process markers"

python3 - "$qual_dir/replay-result.json" "$qual_dir/launcher-result.json" "$run_id" "$gpu" "$scratch_dir" <<'PY'
import json, os, sys
from pathlib import Path
result_path, output_path = map(Path, sys.argv[1:3])
result = json.loads(result_path.read_text())
if result.get("status") != "PASS": raise SystemExit(f"payload result is not PASS: {result.get('error')}")
for name in ("first_pass", "repeat_pass"):
    comparison = result[name]["comparison"]
    if comparison.get("exact_response_token_ids") is not True or comparison.get("samples") != 2:
        raise SystemExit(f"invalid replay comparison: {name}")
record = {
    "container_exit_code": 0,
    "gpu": int(sys.argv[4]),
    "run_id": sys.argv[3],
    "scratch_preserved": sys.argv[5],
    "source_run_id": "20260826T233354Z-skyrl-lifecycle-nccl-dense-r1",
    "status": "PASS",
}
temporary = output_path.with_suffix(".json.tmp")
temporary.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
os.replace(temporary, output_path)
PY

finalize_evidence
trap - EXIT INT TERM
printf 'PASS %s\n' "$run_id"
