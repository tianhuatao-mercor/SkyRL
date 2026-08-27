#!/usr/bin/env bash
set -Eeuo pipefail

readonly EXPECTED_IMAGE_ID="sha256:e98a7978ad815edbd55d460f0f45ec059dcc2df4584e5c8da9c6183b99b2940c"
readonly EXPECTED_IMAGE_REF="skyrl-megatron-b300-cu128-canary:7c528991c4f9-r1"
readonly EXPECTED_SOURCE_REV="7c528991c4f9d470dd9295e10589d99dc3e05053"
readonly EXPECTED_LOCK_SHA="0f3a2126b68747e7d4b854574e9e418c0c4a8f6c9f605865600b04e5d0a2a537"
readonly MODEL_DIR="/shared/models/qwen3-0.6b-c1899de"
readonly MODEL_REV="c1899de289a04d12100db370d81485cdf75e47ca"
readonly DATASET_DIR="/shared/datasets/skyrl-multiply-lifecycle-ebcf5477cdd43cf4"
readonly DATASET_ID="ebcf5477cdd43cf42a380cc0ee168a5c7c4ddcc08521c95d133fa6d5cebaec59"
readonly WORKTREE="/shared/code/SkyRL-b300-lifecycle"
readonly QUAL_ROOT="/shared/environments/b300/qualifications"
readonly CHECKPOINT_ROOT="/shared/checkpoints/qualifications"
readonly RECIPE_ROOT="/shared/environments/b300/recipes"
readonly OWNER_LABEL="skyrl-lifecycle"
readonly HEAD_ALIAS="aws-b300-node1"
readonly WORKER_ALIAS="aws-b300-node2"
readonly HEAD_HOSTNAME="ip-172-31-67-119"
readonly WORKER_HOSTNAME="ip-172-31-67-174"
readonly HEAD_IP="172.31.67.119"
readonly WORKER_IP="172.31.67.174"
readonly HEAD_GPU="2"
readonly WORKER_GPUS="0,1"
readonly RAY_PORT="6387"
readonly DRIVER_RESOURCE="b300_lifecycle_driver"
readonly ROLLOUT_RESOURCE="b300_lifecycle_rollout"
readonly RESUME_SOURCE_RUN_ID="20260827T011849Z-skyrl-lifecycle-2node-2eng-r1"
readonly RESUME_SOURCE_QUAL="$QUAL_ROOT/$RESUME_SOURCE_RUN_ID"
readonly RESUME_SOURCE_CHECKPOINT_ROOT="$CHECKPOINT_ROOT/$RESUME_SOURCE_RUN_ID"
readonly RESUME_SOURCE_STEP="$RESUME_SOURCE_CHECKPOINT_ROOT/checkpoints/global_step_1"
readonly RESUME_SOURCE_EXPORT="$RESUME_SOURCE_CHECKPOINT_ROOT/exports/global_step_1/policy"
readonly RESUME_SOURCE_RESULTS="$RESUME_SOURCE_QUAL/results"
readonly SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=20)

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  printf 'Usage: %s --execute --dataset-dir %s [--resume-from %s]\n' "$0" "$DATASET_DIR" "$RESUME_SOURCE_STEP"
}

execute=false
dataset_dir=""
resume_from=""
while (($#)); do
  case "$1" in
    --execute) execute=true; shift ;;
    --dataset-dir) dataset_dir=${2:?}; shift 2 ;;
    --resume-from) resume_from=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$execute" == true ]] || die "refusing GPU launch without --execute"
[[ "$dataset_dir" == "$DATASET_DIR" ]] || die "dataset must be the exact pinned artifact: $DATASET_DIR"
[[ $(hostname -s) == "$HEAD_HOSTNAME" ]] || die "this gate must be launched from $HEAD_ALIAS"
[[ -d "$WORKTREE/.git" || -f "$WORKTREE/.git" ]] || die "lifecycle worktree missing"

baseline_step=0
final_step=1
training_epochs=1
max_training_steps=1
resume_mode=null
resume_path=null
repeat_eval_step=0
run_suffix="skyrl-lifecycle-2node-2eng-r1"
if [[ -n "$resume_from" ]]; then
  [[ "$resume_from" == "$RESUME_SOURCE_STEP" ]] || die "resume gate requires the exact frozen checkpoint: $RESUME_SOURCE_STEP"
  [[ -d "$RESUME_SOURCE_STEP" && -d "$RESUME_SOURCE_EXPORT" && -d "$RESUME_SOURCE_RESULTS" ]] || die "resume source artifacts are incomplete"
  [[ ! -w "$RESUME_SOURCE_QUAL" && ! -w "$RESUME_SOURCE_CHECKPOINT_ROOT" ]] || die "resume source roots must be read-only"
  baseline_step=1
  final_step=2
  # The source step-1 dataloader snapshot is on the final batch while its
  # iterator remains unfinished. One empty restored outer iteration is allowed
  # before the next real batch; max_training_steps still stops exactly at 2.
  training_epochs=3
  max_training_steps=2
  resume_mode=from_path
  resume_path="$resume_from"
  repeat_eval_step=2
  run_suffix="skyrl-lifecycle-resume-2node-2eng-r1"
fi

node_exec() {
  local host=$1
  shift
  local quoted
  printf -v quoted '%q ' "$@"
  ssh "${SSH_OPTIONS[@]}" "$host" "$quoted"
}

run_id="$(date -u +%Y%m%dT%H%M%SZ)-$run_suffix"
qual_dir="$QUAL_ROOT/$run_id"
checkpoint_root="$CHECKPOINT_ROOT/$run_id"
checkpoint_dir="$checkpoint_root/checkpoints"
export_dir="$checkpoint_root/exports"
result_dir="$qual_dir/results"
infra_log_dir="$qual_dir/inference"
scratch_prefix="/opt/dlami/nvme/cache/tmp/sl/${run_id:0:15}"
head_scratch="${scratch_prefix}-h"
worker_scratch="${scratch_prefix}-w"
head_name="b300-skyrl-$run_id-head"
worker_name="b300-skyrl-$run_id-worker"

for path in "$qual_dir" "$checkpoint_root"; do
  [[ ! -e "$path" ]] || die "refusing to reuse existing run path: $path"
done
mkdir -p "$result_dir" "$infra_log_dir" "$checkpoint_dir" "$export_dir" "$qual_dir/recipe"

head_container_id=""
worker_container_id=""
finalized=false

capture_node_snapshot() {
  local host=$1
  local label=$2
  local prefix="$qual_dir/${label}-${host}"
  node_exec "$host" date -u +%Y-%m-%dT%H:%M:%S.%NZ >"$prefix-timestamp.txt"
  node_exec "$host" hostname -s >"$prefix-hostname.txt"
  node_exec "$host" nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader >"$prefix-gpu-processes.csv" || true
  node_exec "$host" ps -eo comm=,user=,pid=,ppid=,lstart=,args= --sort=pid >"$prefix-processes.txt"
  node_exec "$host" docker ps -a --no-trunc >"$prefix-containers.txt"
  node_exec "$host" docker ps -a --no-trunc --filter "label=io.mercor.qualification.run-id=$run_id" --format '{{.ID}} {{.Names}} {{.Labels}}' >"$prefix-run-containers.txt"
  node_exec "$host" findmnt -T /shared -o TARGET,SOURCE,FSTYPE,OPTIONS >"$prefix-fsx-mount.txt"
  node_exec "$host" ss -lntup >"$prefix-listeners.txt" || true
  node_exec "$host" bash -lc 'ps -eo comm=,args= | awk '\''
    $1 == "raylet" || $1 == "gcs_server" ||
    (($1 == "python" || $1 == "python3") &&
    ($0 ~ /ray::/ || $0 ~ /vllm\.entrypoints/ || $0 ~ /qualification_entrypoint\.py/)) {print}
  '\''' >"$prefix-conflicting-processes.txt"
  jq -n \
    --arg run_id "$run_id" \
    --rawfile timestamp "$prefix-timestamp.txt" \
    --rawfile hostname "$prefix-hostname.txt" \
    --rawfile gpu "$prefix-gpu-processes.csv" \
    --rawfile containers "$prefix-run-containers.txt" \
    --rawfile conflicts "$prefix-conflicting-processes.txt" \
    --rawfile fsx "$prefix-fsx-mount.txt" \
    '{run_id:$run_id,timestamp:($timestamp|rtrimstr("\n")),hostname:($hostname|rtrimstr("\n")),gpu_processes:($gpu|split("\n")|map(select(length>0))),run_owned_containers:($containers|split("\n")|map(select(length>0))),conflicting_processes:($conflicts|split("\n")|map(select(length>0))),fsx_mount:($fsx|rtrimstr("\n"))}' \
    >"$prefix-snapshot.json"
}

verify_container_identity() {
  local host=$1
  local container_id=$2
  local expected_name=$3
  local identity
  identity=$(node_exec "$host" docker inspect "$container_id" --format '{{index .Config.Labels "io.mercor.qualification.run-id"}}|{{index .Config.Labels "io.mercor.qualification.owner"}}|{{.Name}}')
  [[ "$identity" == "$run_id|$OWNER_LABEL|/$expected_name" ]]
}

cleanup_one() {
  local host=$1
  local role=$2
  local container_id=$3
  local expected_name=$4
  [[ -n "$container_id" ]] || return 0
  if node_exec "$host" docker inspect "$container_id" >/dev/null 2>&1; then
    if ! verify_container_identity "$host" "$container_id" "$expected_name"; then
      printf 'ERROR: %s container identity mismatch; refusing cleanup\n' "$role" >&2
      return 1
    fi
    if [[ $(node_exec "$host" docker inspect "$container_id" --format '{{.State.Running}}') == true ]]; then
      node_exec "$host" docker stop --time 60 "$container_id" >"$qual_dir/container-stop-$role.txt"
    fi
    node_exec "$host" docker inspect "$container_id" >"$qual_dir/container-final-$role.json"
    node_exec "$host" docker logs --timestamps "$container_id" >"$qual_dir/container-final-$role.log" 2>&1 || true
    node_exec "$host" docker rm "$container_id" >"$qual_dir/container-remove-$role.txt"
  fi
  [[ -z $(node_exec "$host" docker ps -aq --filter "id=$container_id") ]] || return 1
}

cleanup_all() {
  local rc=0
  cleanup_one "$WORKER_ALIAS" worker "$worker_container_id" "$worker_name" || rc=1
  worker_container_id=""
  cleanup_one "$HEAD_ALIAS" head "$head_container_id" "$head_name" || rc=1
  head_container_id=""
  return "$rc"
}

freeze_evidence() {
  if [[ -d "$checkpoint_root" ]]; then
    (
      cd "$checkpoint_root"
      find . -type f -print0 | sort -z | xargs -0 -r sha256sum
    ) >"$qual_dir/checkpoint-files.sha256" 2>/dev/null || true
  fi
  (
    cd "$qual_dir"
    find . -type f ! -name EVIDENCE.sha256 -print0 | sort -z | xargs -0 -r sha256sum
  ) >"$qual_dir/EVIDENCE.sha256" 2>/dev/null || true
  chmod -R a-w "$qual_dir" "$checkpoint_root" 2>/dev/null || true
}

on_exit() {
  local rc=$?
  if [[ "$finalized" != true ]]; then
    cleanup_all || true
    capture_node_snapshot "$HEAD_ALIAS" post-failure || true
    capture_node_snapshot "$WORKER_ALIAS" post-failure || true
    printf '{"exit_code":%d,"status":"FAIL"}\n' "$rc" >"$qual_dir/launcher-result.json" 2>/dev/null || true
    freeze_evidence
    finalized=true
  fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

if [[ -n "$resume_from" ]]; then
  (
    cd "$RESUME_SOURCE_QUAL"
    sha256sum -c EVIDENCE.sha256
  ) >"$qual_dir/resume-source-evidence-pre.txt"
  (
    cd "$RESUME_SOURCE_CHECKPOINT_ROOT"
    sha256sum -c "$RESUME_SOURCE_QUAL/checkpoint-files.sha256"
  ) >"$qual_dir/resume-source-checkpoint-pre.txt"
  jq -n \
    --arg checkpoint "$RESUME_SOURCE_STEP" \
    --arg export "$RESUME_SOURCE_EXPORT" \
    --arg run_id "$RESUME_SOURCE_RUN_ID" \
    '{checkpoint:$checkpoint,export:$export,run_id:$run_id}' >"$qual_dir/resume-source.json"
  /shared/environments/b300/venvs/skyrl-megatron-0f3a2126/bin/python - \
    "$RESUME_SOURCE_STEP/data.pt" "$qual_dir/resume-boundary-workaround.json" <<'PY'
import json
import os
import sys
from pathlib import Path

import torch

source, output = map(Path, sys.argv[1:])
state = torch.load(source, map_location="cpu", weights_only=False)
record = {
    "iterator_finished": state.get("_iterator_finished"),
    "num_yielded": state.get("_num_yielded"),
    "sampler_iter_yielded": state.get("_sampler_iter_yielded"),
    "samples_yielded": state.get("_sampler_iter_state", {}).get("samples_yielded"),
    "workaround": "allow-one-empty-restored-outer-iteration-before-step-2",
}
expected = {
    "iterator_finished": False,
    "num_yielded": 1,
    "sampler_iter_yielded": 1,
    "samples_yielded": 4,
    "workaround": "allow-one-empty-restored-outer-iteration-before-step-2",
}
if record != expected:
    raise SystemExit(f"unexpected resume boundary state: {record}")
temporary = output.with_suffix(output.suffix + ".tmp")
with temporary.open("x", encoding="utf-8") as stream:
    json.dump(record, stream, sort_keys=True, indent=2)
    stream.write("\n")
    stream.flush()
    os.fsync(stream.fileno())
temporary.replace(output)
PY
fi

/shared/verify-handoff | tee "$qual_dir/verify-handoff.txt"

[[ -z $(git -C "$WORKTREE" status --short) ]] || die "lifecycle worktree must be clean before launch"
git -C "$WORKTREE" merge-base --is-ancestor "$EXPECTED_SOURCE_REV" HEAD || die "qualified source is not an ancestor of the worktree"

for host in "$HEAD_ALIAS" "$WORKER_ALIAS"; do
  node_exec "$host" true
  [[ $(node_exec "$host" docker image inspect "$EXPECTED_IMAGE_REF" --format '{{.Id}}') == "$EXPECTED_IMAGE_ID" ]] || die "image ID drift on $host"
  [[ $(node_exec "$host" docker image inspect "$EXPECTED_IMAGE_ID" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}') == "$EXPECTED_SOURCE_REV" ]] || die "image source revision drift on $host"
  [[ $(node_exec "$host" docker image inspect "$EXPECTED_IMAGE_ID" --format '{{index .Config.Labels "io.mercor.skyrl.lock-sha256"}}') == "$EXPECTED_LOCK_SHA" ]] || die "image lock drift on $host"
  [[ -z $(node_exec "$host" docker ps -aq --filter "label=io.mercor.qualification.owner=$OWNER_LABEL") ]] || die "another lifecycle container exists on $host"
done

head_mount=$(node_exec "$HEAD_ALIAS" findmnt -T /shared -n -o SOURCE,FSTYPE)
worker_mount=$(node_exec "$WORKER_ALIAS" findmnt -T /shared -n -o SOURCE,FSTYPE)
[[ "$head_mount" == "$worker_mount" && "$head_mount" == *"nfs4"* ]] || die "FSx mount drift between the two nodes"

for spec in "$HEAD_ALIAS:$HEAD_GPU" "$WORKER_ALIAS:0" "$WORKER_ALIAS:1"; do
  host=${spec%%:*}
  gpu=${spec##*:}
  used=$(node_exec "$host" nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [[ "$used" == "0" ]] || die "GPU $host:$gpu is not idle: ${used} MiB"
done

capture_node_snapshot "$HEAD_ALIAS" pre
capture_node_snapshot "$WORKER_ALIAS" pre
for host in "$HEAD_ALIAS" "$WORKER_ALIAS"; do
  jq -e '.gpu_processes|length==0' "$qual_dir/pre-$host-snapshot.json" >/dev/null || die "GPU processes exist on $host"
  jq -e '.run_owned_containers|length==0' "$qual_dir/pre-$host-snapshot.json" >/dev/null || die "run-owned containers exist on $host"
  jq -e '.conflicting_processes|length==0' "$qual_dir/pre-$host-snapshot.json" >/dev/null || die "Ray/vLLM/SkyRL processes exist on $host"
done

[[ -f "$MODEL_DIR/MODEL_MANIFEST.json" ]] || die "model manifest missing"
[[ -f "$DATASET_DIR/manifest.json" ]] || die "dataset manifest missing"
/shared/environments/b300/venvs/skyrl-megatron-0f3a2126/bin/python \
  "$WORKTREE/platform/b300/lifecycle/validate_pinned_inputs.py" \
  --model-dir "$MODEL_DIR" \
  --model-revision "$MODEL_REV" \
  --dataset-dir "$DATASET_DIR" \
  --dataset-id "$DATASET_ID" \
  --model-output "$qual_dir/model-validation.json" \
  --dataset-output "$qual_dir/dataset-validation.json"

runtime_files=(
  platform/b300/lifecycle/qualification_entrypoint.py
  skyrl/backends/skyrl_train/inference_servers/common.py
  skyrl/backends/skyrl_train/inference_servers/utils.py
  skyrl/backends/skyrl_train/inference_servers/vllm_router.py
  skyrl/backends/skyrl_train/inference_servers/vllm_server_actor.py
  skyrl/backends/skyrl_train/weight_sync/broadcast_strategy.py
  skyrl/train/utils/utils.py
)
payload_hash=$(
  for relative in "${runtime_files[@]}"; do sha256sum "$WORKTREE/$relative" | awk '{print $1}'; done |
    sha256sum | awk '{print $1}'
)
payload_dir="$RECIPE_ROOT/skyrl-lifecycle-$payload_hash"
if [[ ! -e "$payload_dir" ]]; then
  mkdir "$payload_dir"
  for relative in "${runtime_files[@]}"; do
    mkdir -p "$payload_dir/$(dirname "$relative")"
    install -m 0444 "$WORKTREE/$relative" "$payload_dir/$relative"
  done
  (
    cd "$payload_dir"
    sha256sum "${runtime_files[@]}" >PAYLOAD.sha256
  )
  chmod 0444 "$payload_dir/PAYLOAD.sha256"
  find "$payload_dir" -type d -exec chmod 0555 {} +
else
  (cd "$payload_dir" && sha256sum -c PAYLOAD.sha256)
fi

install -m 0444 "$WORKTREE/platform/b300/lifecycle/"{qualification_entrypoint.py,run_multinode_lifecycle.sh,validate_pinned_inputs.py,verify_artifacts.py,verify_multinode_artifacts.py,verify_resume_artifacts.py} "$qual_dir/recipe/"
git -C "$WORKTREE" status --short >"$qual_dir/worktree-status.txt"
git -C "$WORKTREE" rev-parse HEAD >"$qual_dir/worktree-revision.txt"
git -C "$WORKTREE" remote -v >"$qual_dir/worktree-remotes.txt"
git -C "$WORKTREE" diff --binary >"$qual_dir/worktree.patch"
docker image inspect "$EXPECTED_IMAGE_ID" >"$qual_dir/image-inspect.json"
cp "$MODEL_DIR/MODEL_MANIFEST.json" "$qual_dir/model-manifest.json"
cp "$DATASET_DIR/manifest.json" "$qual_dir/dataset-manifest.json"
cp "$payload_dir/PAYLOAD.sha256" "$qual_dir/payload.sha256"

cmd=(
  /opt/venvs/skyrl-megatron/bin/python /opt/qualification/qualification_entrypoint.py
  "data.train_data=[\"$DATASET_DIR/train.parquet\"]"
  "data.val_data=[\"$DATASET_DIR/validation.parquet\"]"
  data.dataloader.num_workers=0
  data.dataloader.persistent_workers=false
  trainer.seed=42
  trainer.strategy=megatron
  trainer.policy.model.path="$MODEL_DIR"
  trainer.critic.model.path=null
  trainer.policy.model.lora.rank=0
  trainer.placement.colocate_all=false
  trainer.placement.colocate_policy_ref=false
  trainer.placement.policy_num_nodes=1
  trainer.placement.policy_num_gpus_per_node=1
  trainer.policy.megatron_config.tensor_model_parallel_size=1
  trainer.policy.megatron_config.pipeline_model_parallel_size=1
  trainer.policy.megatron_config.context_parallel_size=1
  trainer.use_expandable_segments=true
  trainer.flash_attn=true
  trainer.gradient_checkpointing=true
  trainer.remove_microbatch_padding=true
  trainer.algorithm.advantage_estimator=grpo
  trainer.algorithm.use_kl_loss=false
  trainer.algorithm.use_kl_in_reward=false
  trainer.algorithm.temperature=0.8
  trainer.epochs="$training_epochs"
  trainer.max_training_steps="$max_training_steps"
  trainer.update_epochs_per_batch=1
  trainer.train_batch_size=4
  trainer.policy_mini_batch_size=4
  trainer.micro_train_batch_size_per_gpu=4
  trainer.micro_forward_batch_size_per_gpu=4
  trainer.max_prompt_length=128
  trainer.policy.optimizer_config.lr=1.0e-4
  trainer.policy.optimizer_config.max_grad_norm=1.0
  trainer.policy.optimizer_config.offload_after_step=false
  trainer.eval_batch_size=2
  trainer.eval_before_train=true
  trainer.eval_interval=1
  trainer.ckpt_interval=1
  trainer.hf_save_interval=1
  trainer.max_ckpts_to_keep=-1
  trainer.resume_mode="$resume_mode"
  trainer.resume_path="$resume_path"
  trainer.ckpt_path="$checkpoint_dir"
  trainer.export_path="$export_dir"
  trainer.log_path="$infra_log_dir"
  trainer.dump_data_batch=true
  trainer.dump_eval_results=true
  trainer.num_logger_train_samples=16
  trainer.num_logger_eval_samples=2
  trainer.print_example_interval=1
  trainer.logger=console
  trainer.project_name=b300-skyrl-lifecycle
  trainer.run_name="$run_id"
  trainer.enable_ray_gpu_monitor=false
  generator.inference_engine.backend=vllm
  generator.inference_engine.run_engines_locally=true
  generator.inference_engine.num_engines=2
  generator.inference_engine.tensor_parallel_size=1
  generator.inference_engine.pipeline_parallel_size=1
  generator.inference_engine.data_parallel_size=1
  generator.inference_engine.weight_sync_backend=nccl
  generator.inference_engine.served_model_name=b300-qwen3-0.6b-c1899de
  generator.inference_engine.model_dtype=bfloat16
  generator.inference_engine.gpu_memory_utilization=0.15
  generator.inference_engine.enforce_eager=true
  generator.inference_engine.enable_prefix_caching=false
  generator.inference_engine.offload_kv_for_weight_sync=false
  generator.inference_engine.use_expandable_segments=false
  generator.inference_engine.enable_ray_prometheus_stats=false
  generator.inference_engine.max_num_seqs=4
  generator.inference_engine.max_num_batched_tokens=1024
  generator.inference_engine.engine_init_kwargs.max_model_len=1024
  generator.inference_engine.distributed_executor_backend=ray
  generator.batched=false
  generator.max_turns=1
  generator.max_input_length=128
  generator.chat_template_kwargs.enable_thinking=false
  generator.n_samples_per_prompt=4
  generator.sampling_params.max_generate_length=128
  generator.sampling_params.temperature=0.8
  generator.sampling_params.top_p=1.0
  generator.sampling_params.top_k=-1
  generator.sampling_params.logprobs=1
  generator.eval_n_samples_per_prompt=1
  generator.eval_sampling_params.max_generate_length=128
  generator.eval_sampling_params.temperature=0.0
  generator.eval_sampling_params.top_p=1.0
  generator.eval_sampling_params.top_k=-1
  generator.eval_sampling_params.logprobs=1
  generator.trajectory_max_attempts=1
  generator.generation_batch_timeout_s=300
  environment.env_class=multiply
  environment.skyrl_gym.max_env_workers=4
)
printf '%s\0' "${cmd[@]}" | jq -Rs 'split("\u0000")[:-1]' >"$qual_dir/command.json"

PYTHONPATH="$WORKTREE" \
PYTHONDONTWRITEBYTECODE=1 \
SKYRL_QUAL_RESULT_DIR="$result_dir" \
SKYRL_CONFIG_PREFLIGHT_ONLY=1 \
  /shared/environments/b300/venvs/skyrl-megatron-0f3a2126/bin/python \
  "$qual_dir/recipe/qualification_entrypoint.py" "${cmd[@]:2}"

for spec in "$HEAD_ALIAS:$head_scratch" "$WORKER_ALIAS:$worker_scratch"; do
  host=${spec%%:*}
  scratch=${spec#*:}
  [[ $(node_exec "$host" bash -lc 'test -w /opt/dlami/nvme/cache/tmp && printf yes') == yes ]] || die "NVMe cache root is not writable on $host"
  node_exec "$host" test ! -e "$scratch" || die "scratch path already exists on $host: $scratch"
  node_exec "$host" mkdir -p "$scratch"/{h-head,h-driver,t,r,xdg,triton,vllm}
done

mapfile -t infiniband_devices < <(find /dev/infiniband -maxdepth 1 -type c -print | sort)
(( ${#infiniband_devices[@]} > 0 )) || die "no EFA device nodes found"
for device in "${infiniband_devices[@]}"; do
  node_exec "$WORKER_ALIAS" test -c "$device" || die "EFA device missing on worker: $device"
done

create_container() {
  local host=$1
  local role=$2
  local name=$3
  local gpu_spec=$4
  local scratch=$5
  shift 5
  local role_home=/c/h-head
  local -a host_gpu_ids=()
  IFS=, read -r -a host_gpu_ids <<<"$gpu_spec"
  local cuda_visible
  cuda_visible=$(seq -s, 0 $(( ${#host_gpu_ids[@]} - 1 )))
  [[ "$role" != driver ]] || role_home=/c/h-driver
  local -a args=(
    docker create
    --pull never
    --name "$name"
    --label "io.mercor.qualification.run-id=$run_id"
    --label "io.mercor.qualification.owner=$OWNER_LABEL"
    --label "io.mercor.qualification.role=$role"
    --init
    --stop-timeout 60
    --gpus "\"device=$gpu_spec\""
    --network host
    --ipc host
    --hostname "$(node_exec "$host" hostname -s)"
    --user "$(id -u):$(id -g)"
    --ulimit memlock=-1:-1
    --ulimit stack=67108864:67108864
    --ulimit nofile=1048576:1048576
    --mount type=bind,src=/shared,dst=/shared,readonly
    --mount "type=bind,src=$qual_dir,dst=$qual_dir"
    --mount "type=bind,src=$checkpoint_root,dst=$checkpoint_root"
    --mount "type=bind,src=$scratch,dst=/c"
    --mount "type=bind,src=$payload_dir/platform/b300/lifecycle/qualification_entrypoint.py,dst=/opt/qualification/qualification_entrypoint.py,readonly"
    --mount "type=bind,src=$payload_dir/skyrl/backends/skyrl_train/inference_servers/common.py,dst=/workspace/SkyRL/skyrl/backends/skyrl_train/inference_servers/common.py,readonly"
    --mount "type=bind,src=$payload_dir/skyrl/backends/skyrl_train/inference_servers/utils.py,dst=/workspace/SkyRL/skyrl/backends/skyrl_train/inference_servers/utils.py,readonly"
    --mount "type=bind,src=$payload_dir/skyrl/backends/skyrl_train/inference_servers/vllm_router.py,dst=/workspace/SkyRL/skyrl/backends/skyrl_train/inference_servers/vllm_router.py,readonly"
    --mount "type=bind,src=$payload_dir/skyrl/backends/skyrl_train/inference_servers/vllm_server_actor.py,dst=/workspace/SkyRL/skyrl/backends/skyrl_train/inference_servers/vllm_server_actor.py,readonly"
    --mount "type=bind,src=$payload_dir/skyrl/backends/skyrl_train/weight_sync/broadcast_strategy.py,dst=/workspace/SkyRL/skyrl/backends/skyrl_train/weight_sync/broadcast_strategy.py,readonly"
    --mount "type=bind,src=$payload_dir/skyrl/train/utils/utils.py,dst=/workspace/SkyRL/skyrl/train/utils/utils.py,readonly"
    --env "NVIDIA_VISIBLE_DEVICES=$gpu_spec"
    --env "CUDA_VISIBLE_DEVICES=$cuda_visible"
    --env HF_HUB_OFFLINE=1
    --env TRANSFORMERS_OFFLINE=1
    --env HF_DATASETS_OFFLINE=1
    --env HF_HUB_DISABLE_TELEMETRY=1
    --env VLLM_NO_USAGE_STATS=1
    --env VLLM_USE_FLASHINFER_SAMPLER=0
    --env RAY_USAGE_STATS_ENABLED=0
    --env WANDB_MODE=offline
    --env DO_NOT_TRACK=1
    --env PYTHONDONTWRITEBYTECODE=1
    --env PYTHONNOUSERSITE=1
    --env CUDA_DEVICE_MAX_CONNECTIONS=1
    --env NCCL_CUMEM_ENABLE=0
    --env NCCL_DEBUG=INFO
    --env NCCL_DEBUG_SUBSYS=INIT,NET,ENV
    --env "NCCL_DEBUG_FILE=$qual_dir/nccl-%h-%p.log"
    --env "HOME=$role_home"
    --env TMPDIR=/c/t
    --env RAY_TMPDIR=/c/r
    --env XDG_CACHE_HOME=/c/xdg
    --env TRITON_CACHE_DIR=/c/triton
    --env VLLM_CACHE_ROOT=/c/vllm
  )
  for device in "${infiniband_devices[@]}"; do
    args+=(--device "$device:$device")
  done
  args+=("$EXPECTED_IMAGE_ID" "$@")
  printf '%s\0' "${args[@]}" | jq -Rs 'split("\u0000")[:-1]' >"$qual_dir/docker-create-$role.json"
  node_exec "$host" "${args[@]}"
}

head_container_id=$(create_container \
  "$HEAD_ALIAS" head "$head_name" "$HEAD_GPU" "$head_scratch" \
  /opt/venvs/skyrl-megatron/bin/ray start \
  --head \
  "--node-ip-address=$HEAD_IP" \
  "--port=$RAY_PORT" \
  --num-gpus=1 \
  --num-cpus=32 \
  "--resources={\"$DRIVER_RESOURCE\":1}" \
  --disable-usage-stats \
  --dashboard-host=127.0.0.1 \
  --temp-dir=/c/r/head \
  --block)
verify_container_identity "$HEAD_ALIAS" "$head_container_id" "$head_name" || die "head container identity mismatch"
node_exec "$HEAD_ALIAS" docker inspect "$head_container_id" >"$qual_dir/container-prestart-head.json"
node_exec "$HEAD_ALIAS" docker start "$head_container_id" >"$qual_dir/container-start-head.txt"
node_exec "$HEAD_ALIAS" bash -lc "for i in \$(seq 1 60); do if timeout 1 bash -c '</dev/tcp/$HEAD_IP/$RAY_PORT' 2>/dev/null; then exit 0; fi; sleep 1; done; exit 1" || die "Ray head did not accept private-IP connections within 60 seconds"
node_exec "$HEAD_ALIAS" docker exec "$head_container_id" \
  /opt/venvs/skyrl-megatron/bin/python -c \
  'import json, os, torch; count=torch.cuda.device_count(); print(json.dumps({"cuda_available":torch.cuda.is_available(),"cuda_device_count":count,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"devices":[torch.cuda.get_device_name(i) for i in range(count)],"status":"PASS" if count == 1 else "FAIL"},sort_keys=True)); assert count == 1' \
  >"$qual_dir/head-cuda-visibility.json"

worker_container_id=$(create_container \
  "$WORKER_ALIAS" worker "$worker_name" "$WORKER_GPUS" "$worker_scratch" \
  /opt/venvs/skyrl-megatron/bin/ray start \
  "--address=$HEAD_IP:$RAY_PORT" \
  "--node-ip-address=$WORKER_IP" \
  --num-gpus=2 \
  --num-cpus=32 \
  "--resources={\"$ROLLOUT_RESOURCE\":1}" \
  --disable-usage-stats \
  --block)
verify_container_identity "$WORKER_ALIAS" "$worker_container_id" "$worker_name" || die "worker container identity mismatch"
node_exec "$WORKER_ALIAS" docker inspect "$worker_container_id" >"$qual_dir/container-prestart-worker.json"
node_exec "$WORKER_ALIAS" docker start "$worker_container_id" >"$qual_dir/container-start-worker.txt"

ray_probe_code="$(printf '%s\n' \
  'import json, ray' \
  "ray.init(address='$HEAD_IP:$RAY_PORT', log_to_driver=False, logging_level='ERROR')" \
  'nodes = {node["NodeManagerAddress"]: node for node in ray.nodes() if node["Alive"]}' \
  "assert set(nodes) == {'$HEAD_IP', '$WORKER_IP'}, sorted(nodes)" \
  "assert nodes['$HEAD_IP']['Resources'].get('GPU') == 1" \
  "assert nodes['$HEAD_IP']['Resources'].get('$DRIVER_RESOURCE', 0) >= 1" \
  "assert nodes['$WORKER_IP']['Resources'].get('GPU') == 2" \
  "assert nodes['$WORKER_IP']['Resources'].get('$ROLLOUT_RESOURCE', 0) >= 1" \
  'print(json.dumps({"nodes": sorted(nodes), "status": "PASS"}, sort_keys=True))' \
  'ray.shutdown()')"
worker_ready=false
for _ in $(seq 1 20); do
  if node_exec "$HEAD_ALIAS" docker exec "$head_container_id" \
    /opt/venvs/skyrl-megatron/bin/python -c "$ray_probe_code" \
    >>"$qual_dir/ray-cluster-readiness-attempts.log" 2>&1; then
    worker_ready=true
    break
  fi
  [[ $(node_exec "$WORKER_ALIAS" docker inspect "$worker_container_id" --format '{{.State.Running}}') == true ]] || break
  sleep 1
done
[[ "$worker_ready" == true ]] || die "Ray worker did not register with the exact role resources"
node_exec "$WORKER_ALIAS" docker exec "$worker_container_id" \
  /opt/venvs/skyrl-megatron/bin/python -c \
  'import json, os, torch; count=torch.cuda.device_count(); print(json.dumps({"cuda_available":torch.cuda.is_available(),"cuda_device_count":count,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"devices":[torch.cuda.get_device_name(i) for i in range(count)],"status":"PASS" if count == 2 else "FAIL"},sort_keys=True)); assert count == 2' \
  >"$qual_dir/worker-cuda-visibility.json"

node_exec "$HEAD_ALIAS" docker exec "$head_container_id" /opt/venvs/skyrl-megatron/bin/ray status "--address=$HEAD_IP:$RAY_PORT" >"$qual_dir/ray-status-pre-driver.txt"

driver_exec=(
  docker exec
  "$head_container_id"
  env \
  "HOME=/c/h-driver" \
  "RAY_ADDRESS=$HEAD_IP:$RAY_PORT" \
  "SKYRL_INFERENCE_BIND_HOST=ray-node-ip" \
  "SKYRL_INFERENCE_ADVERTISE_HOST=ray-node-ip" \
  "SKYRL_WEIGHT_SYNC_MASTER_ADDR=$HEAD_IP" \
  "SKYRL_QUAL_RESULT_DIR=$result_dir" \
  "SKYRL_QUAL_REPEAT_EVAL_STEP=$repeat_eval_step" \
  "SKYRL_QUAL_DRIVER_RESOURCE=$DRIVER_RESOURCE" \
  "SKYRL_QUAL_ROLLOUT_RESOURCE=$ROLLOUT_RESOURCE" \
  "SKYRL_QUAL_EXPECTED_RAY_NODE_IPS=$HEAD_IP,$WORKER_IP" \
  SKYRL_RAY_PG_TIMEOUT_IN_S=180 \
  "${cmd[@]}"
)
printf '%s\0' "${driver_exec[@]}" | jq -Rs 'split("\u0000")[:-1]' >"$qual_dir/docker-exec-driver.json"
node_exec "$HEAD_ALIAS" docker inspect "$head_container_id" >"$qual_dir/container-pre-driver-head.json"
printf -v driver_remote_command '%q ' "${driver_exec[@]}"

set +e
timeout --foreground --signal=TERM --kill-after=60s 1800s \
  ssh "${SSH_OPTIONS[@]}" "$HEAD_ALIAS" "$driver_remote_command" 2>&1 |
  tee "$qual_dir/attached.log"
driver_start_rc=${PIPESTATUS[0]}
set -e

cleanup_all
capture_node_snapshot "$HEAD_ALIAS" post
capture_node_snapshot "$WORKER_ALIAS" post

[[ "$driver_start_rc" -eq 0 ]] || die "bounded driver failed or timed out: rc=$driver_start_rc"

for host in "$HEAD_ALIAS" "$WORKER_ALIAS"; do
  jq -e '.gpu_processes|length==0' "$qual_dir/post-$host-snapshot.json" >/dev/null || die "GPU processes remain on $host"
  jq -e '.run_owned_containers|length==0' "$qual_dir/post-$host-snapshot.json" >/dev/null || die "run-owned containers remain on $host"
  jq -e '.conflicting_processes|length==0' "$qual_dir/post-$host-snapshot.json" >/dev/null || die "Ray/vLLM/SkyRL processes remain on $host"
done

awk '1' "$qual_dir/post-$HEAD_ALIAS-gpu-processes.csv" "$qual_dir/post-$WORKER_ALIAS-gpu-processes.csv" >"$qual_dir/post-selected-gpu-processes.csv"
awk '1' "$qual_dir/post-$HEAD_ALIAS-containers.txt" "$qual_dir/post-$WORKER_ALIAS-containers.txt" >"$qual_dir/post-containers-combined.txt"
awk '1' "$qual_dir/post-$HEAD_ALIAS-processes.txt" "$qual_dir/post-$WORKER_ALIAS-processes.txt" >"$qual_dir/post-processes-combined.txt"

if [[ -n "$resume_from" ]]; then
  (
    cd "$RESUME_SOURCE_QUAL"
    sha256sum -c EVIDENCE.sha256
  ) >"$qual_dir/resume-source-evidence-post.txt"
  (
    cd "$RESUME_SOURCE_CHECKPOINT_ROOT"
    sha256sum -c "$RESUME_SOURCE_QUAL/checkpoint-files.sha256"
  ) >"$qual_dir/resume-source-checkpoint-post.txt"

  /shared/environments/b300/venvs/skyrl-megatron-0f3a2126/bin/python \
    "$qual_dir/recipe/verify_resume_artifacts.py" \
    --result-dir "$result_dir" \
    --checkpoint-dir "$checkpoint_dir" \
    --export-dir "$export_dir" \
    --source-result-dir "$RESUME_SOURCE_RESULTS" \
    --source-export-dir "$RESUME_SOURCE_EXPORT" \
    --resume-checkpoint "$RESUME_SOURCE_STEP" \
    --source-evidence-pre "$qual_dir/resume-source-evidence-pre.txt" \
    --source-evidence-post "$qual_dir/resume-source-evidence-post.txt" \
    --source-checkpoint-pre "$qual_dir/resume-source-checkpoint-pre.txt" \
    --source-checkpoint-post "$qual_dir/resume-source-checkpoint-post.txt" \
    --resume-boundary-record "$qual_dir/resume-boundary-workaround.json" \
    --attached-log "$qual_dir/attached.log" \
    --post-gpu-processes "$qual_dir/post-selected-gpu-processes.csv" \
    --post-containers "$qual_dir/post-containers-combined.txt" \
    --post-processes "$qual_dir/post-processes-combined.txt" \
    --run-id "$run_id" \
    --expected-inference-receivers 2 \
    --expected-inference-host "$WORKER_IP" \
    --inference-log-dir "$infra_log_dir" \
    2> >(tee "$qual_dir/artifact-verification.stderr" >&2) |
    tee "$qual_dir/artifact-verification.stdout"

  (
    cd "$RESUME_SOURCE_QUAL"
    sha256sum -c EVIDENCE.sha256
  ) >"$qual_dir/resume-source-evidence-final.txt"
  (
    cd "$RESUME_SOURCE_CHECKPOINT_ROOT"
    sha256sum -c "$RESUME_SOURCE_QUAL/checkpoint-files.sha256"
  ) >"$qual_dir/resume-source-checkpoint-final.txt"
  cmp -s "$qual_dir/resume-source-evidence-pre.txt" "$qual_dir/resume-source-evidence-final.txt" || die "source evidence changed during resume verification"
  cmp -s "$qual_dir/resume-source-checkpoint-pre.txt" "$qual_dir/resume-source-checkpoint-final.txt" || die "source checkpoint changed during resume verification"
else
  /shared/environments/b300/venvs/skyrl-megatron-0f3a2126/bin/python \
    "$qual_dir/recipe/verify_artifacts.py" \
    --result-dir "$result_dir" \
    --checkpoint-dir "$checkpoint_dir" \
    --export-dir "$export_dir" \
    --attached-log "$qual_dir/attached.log" \
    --post-gpu-processes "$qual_dir/post-selected-gpu-processes.csv" \
    --post-containers "$qual_dir/post-containers-combined.txt" \
    --post-processes "$qual_dir/post-processes-combined.txt" \
    --run-id "$run_id" \
    --expected-inference-receivers 2 \
    --expected-inference-host "$WORKER_IP" \
    --inference-log-dir "$infra_log_dir" \
    2> >(tee "$qual_dir/artifact-verification.stderr" >&2) |
    tee "$qual_dir/artifact-verification.stdout"
fi

/shared/environments/b300/venvs/skyrl-megatron-0f3a2126/bin/python \
  "$qual_dir/recipe/verify_multinode_artifacts.py" \
  --result-dir "$result_dir" \
  --nccl-log-dir "$qual_dir" \
  --head-ip "$HEAD_IP" \
  --worker-ip "$WORKER_IP" \
  --head-hostname "$HEAD_HOSTNAME" \
  --worker-hostname "$WORKER_HOSTNAME" \
  --driver-resource "$DRIVER_RESOURCE" \
  --rollout-resource "$ROLLOUT_RESOURCE" \
  --post-head-snapshot "$qual_dir/post-$HEAD_ALIAS-snapshot.json" \
  --post-worker-snapshot "$qual_dir/post-$WORKER_ALIAS-snapshot.json" \
  --run-id "$run_id" \
  --output "$result_dir/multinode-artifact-verification.json" \
  2> >(tee "$qual_dir/multinode-verification.stderr" >&2) |
  tee "$qual_dir/multinode-verification.stdout"

jq -n \
  --argjson baseline_step "$baseline_step" \
  --argjson final_step "$final_step" \
  --arg run_id "$run_id" \
  --arg head "$HEAD_ALIAS:$HEAD_GPU" \
  --arg worker "$WORKER_ALIAS:$WORKER_GPUS" \
  --arg head_scratch "$head_scratch" \
  --arg resume_from "$resume_from" \
  --arg worker_scratch "$worker_scratch" \
  '{baseline_step:$baseline_step,driver_exit_code:0,driver_gpu:$head,final_step:$final_step,head_scratch_preserved:$head_scratch,inference_gpus:$worker,num_engines:2,num_nodes:2,resume_from:($resume_from | if length == 0 then null else . end),run_id:$run_id,status:"PASS",worker_scratch_preserved:$worker_scratch}' \
  >"$qual_dir/launcher-result.json"

freeze_evidence
finalized=true
trap - EXIT INT TERM
printf 'PASS %s\n' "$run_id"
