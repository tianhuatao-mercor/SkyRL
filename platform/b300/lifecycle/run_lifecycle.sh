#!/usr/bin/env bash
set -Eeuo pipefail

readonly EXPECTED_IMAGE_ID="sha256:e98a7978ad815edbd55d460f0f45ec059dcc2df4584e5c8da9c6183b99b2940c"
readonly EXPECTED_IMAGE_REF="skyrl-megatron-b300-cu128-canary:7c528991c4f9-r1"
readonly EXPECTED_SOURCE_REV="7c528991c4f9d470dd9295e10589d99dc3e05053"
readonly EXPECTED_LOCK_SHA="0f3a2126b68747e7d4b854574e9e418c0c4a8f6c9f605865600b04e5d0a2a537"
readonly MODEL_DIR="/shared/models/qwen3-0.6b-c1899de"
readonly MODEL_REV="c1899de289a04d12100db370d81485cdf75e47ca"
readonly EXPECTED_DATASET_DIR="/shared/datasets/skyrl-multiply-lifecycle-ebcf5477cdd43cf4"
readonly EXPECTED_DATASET_ID="ebcf5477cdd43cf42a380cc0ee168a5c7c4ddcc08521c95d133fa6d5cebaec59"
readonly OWNER_LABEL="skyrl-lifecycle"
readonly QUAL_ROOT="/shared/environments/b300/qualifications"
readonly CHECKPOINT_ROOT="/shared/checkpoints/qualifications"
readonly RECIPE_ROOT="/shared/environments/b300/recipes"
readonly WORKTREE="/shared/code/SkyRL-b300-lifecycle"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  printf 'Usage: %s --execute --dataset-dir /shared/datasets/skyrl-multiply-lifecycle-<id> [--gpus 0,1]\n' "$0"
}

execute=false
dataset_dir=""
gpu_ids="0,1"
while (($#)); do
  case "$1" in
    --execute) execute=true; shift ;;
    --dataset-dir) dataset_dir=${2:?}; shift 2 ;;
    --gpus) gpu_ids=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$execute" == true ]] || die "refusing GPU launch without --execute"
[[ "$dataset_dir" == "$EXPECTED_DATASET_DIR" ]] || die "dataset must be the exact pinned artifact: $EXPECTED_DATASET_DIR"
[[ "$gpu_ids" =~ ^[0-7],[0-7]$ ]] || die "--gpus must select exactly two B300 indices"
IFS=, read -r rollout_gpu policy_gpu <<<"$gpu_ids"
[[ "$rollout_gpu" != "$policy_gpu" ]] || die "two distinct GPU indices are required"
[[ $(hostname -s) == "ip-172-31-67-119" ]] || die "this bounded gate is pinned to aws-b300-node1"
[[ -f "$dataset_dir/manifest.json" ]] || die "dataset manifest missing"
[[ -f "$MODEL_DIR/MODEL_MANIFEST.json" ]] || die "model manifest missing"
[[ -d "$WORKTREE/.git" || -f "$WORKTREE/.git" ]] || die "lifecycle worktree missing"

actual_image_id=$(docker image inspect "$EXPECTED_IMAGE_REF" --format '{{.Id}}')
[[ "$actual_image_id" == "$EXPECTED_IMAGE_ID" ]] || die "image ID drift: $actual_image_id"
[[ $(docker image inspect "$EXPECTED_IMAGE_ID" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}') == "$EXPECTED_SOURCE_REV" ]] || die "image source revision label drift"
[[ $(docker image inspect "$EXPECTED_IMAGE_ID" --format '{{index .Config.Labels "io.mercor.skyrl.lock-sha256"}}') == "$EXPECTED_LOCK_SHA" ]] || die "image lock label drift"
[[ $(docker image inspect "$EXPECTED_IMAGE_ID" --format '{{index .Config.Labels "io.mercor.stack.torch-nccl"}}') == "2.28.9" ]] || die "image torch NCCL label drift"
[[ $(docker image inspect "$EXPECTED_IMAGE_ID" --format '{{index .Config.Labels "io.mercor.stack.vllm"}}') == "0.26.0" ]] || die "image vLLM label drift"
[[ $(git -C "$WORKTREE" rev-parse HEAD) == "$EXPECTED_SOURCE_REV" ]] || die "worktree base revision drift"

run_id="$(date -u +%Y%m%dT%H%M%SZ)-skyrl-lifecycle-nccl-dense-r1"
qual_dir="$QUAL_ROOT/$run_id"
checkpoint_dir="$CHECKPOINT_ROOT/$run_id/checkpoints"
export_dir="$CHECKPOINT_ROOT/$run_id/exports"
result_dir="$qual_dir/results"
infra_log_dir="$qual_dir/inference"
scratch_dir="/opt/dlami/nvme/cache/l/${run_id:0:15}"
container_name="b300-skyrl-$run_id"

for path in "$qual_dir" "$CHECKPOINT_ROOT/$run_id" "$scratch_dir"; do
  [[ ! -e "$path" ]] || die "refusing to reuse existing run path: $path"
done
[[ -z $(docker ps -aq --filter "name=^/${container_name}$") ]] || die "container name already exists"

mkdir -p "$result_dir" "$infra_log_dir" "$checkpoint_dir" "$export_dir" "$scratch_dir"/{h,t,r,xdg,triton,vllm}

container_id=""
finalized=false

snapshot() {
  local prefix=$1
  date -u +%Y-%m-%dT%H:%M:%S.%NZ >"$qual_dir/${prefix}-timestamp.txt"
  nvidia-smi -q -x >"$qual_dir/${prefix}-nvidia-smi.xml"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader >"$qual_dir/${prefix}-gpu-processes.csv" || true
  ps -eo user,pid,ppid,lstart,args --sort=pid >"$qual_dir/${prefix}-processes.txt"
  docker ps -a --no-trunc >"$qual_dir/${prefix}-containers.txt"
  findmnt -T /shared -o TARGET,SOURCE,FSTYPE,OPTIONS >"$qual_dir/${prefix}-fsx-mount.txt"
  ss -lntup >"$qual_dir/${prefix}-listeners.txt" || true
}

verify_container_identity() {
  local captured_id=$1
  [[ $(docker inspect "$captured_id" --format '{{index .Config.Labels "io.mercor.qualification.run-id"}}') == "$run_id" ]]
  [[ $(docker inspect "$captured_id" --format '{{index .Config.Labels "io.mercor.qualification.owner"}}') == "$OWNER_LABEL" ]]
  [[ $(docker inspect "$captured_id" --format '{{.Name}}') == "/$container_name" ]]
}

cleanup_owned_container() {
  [[ -n "$container_id" ]] || return 0
  if docker inspect "$container_id" >/dev/null 2>&1; then
    if ! verify_container_identity "$container_id"; then
      printf 'ERROR: captured container identity mismatch; refusing cleanup\n' >&2
      return 1
    fi
    if [[ $(docker inspect "$container_id" --format '{{.State.Running}}') == true ]]; then
      docker stop --time 60 "$container_id" >"$qual_dir/container-stop.txt"
    fi
    docker inspect "$container_id" >"$qual_dir/container-final-inspect.json"
    docker logs --timestamps "$container_id" >"$qual_dir/container-final.log" 2>&1 || true
    docker rm "$container_id" >"$qual_dir/container-remove.txt"
  fi
  if [[ -n $(docker ps -aq --filter "id=$container_id") ]]; then
    printf 'ERROR: captured container still exists after removal\n' >&2
    return 1
  fi
  container_id=""
}

on_exit() {
  local rc=$?
  if [[ "$finalized" != true ]]; then
    cleanup_owned_container || true
    snapshot post-failure || true
    printf '{"exit_code":%d,"status":"FAIL"}\n' "$rc" >"$qual_dir/launcher-result.json" 2>/dev/null || true
  fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

snapshot pre

for gpu in "$rollout_gpu" "$policy_gpu"; do
  used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
  [[ "$used" == "0" ]] || die "GPU $gpu is not idle: ${used} MiB used"
  gpu_uuid=$(nvidia-smi --id="$gpu" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
  if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F, -v uuid="$gpu_uuid" '$1 == uuid {found=1} END {exit !found}'; then
    die "GPU $gpu has an active compute process"
  fi
done

ps -eo comm=,args= | awk '
  $1 == "raylet" || $1 == "gcs_server" ||
  $0 ~ /ray::/ || $0 ~ /vllm\.entrypoints/ || $0 ~ /qualification_entrypoint\.py/ {print}
' >"$qual_dir/pre-conflicting-processes.txt"
[[ ! -s "$qual_dir/pre-conflicting-processes.txt" ]] || die "conflicting Ray/vLLM/SkyRL processes exist before launch"
[[ -z $(docker ps -aq --filter "label=io.mercor.qualification.owner=$OWNER_LABEL") ]] || die "another lifecycle container exists"

python3 - "$MODEL_DIR/MODEL_MANIFEST.json" "$MODEL_REV" "$qual_dir/model-validation.json" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
manifest_path, expected_revision, output = map(Path, (sys.argv[1], sys.argv[2], sys.argv[3]))
manifest = json.loads(manifest_path.read_text())
if manifest["resolved_revision"] != str(expected_revision):
    raise SystemExit("model revision mismatch")
for record in manifest["files"]:
    path = manifest_path.parent / record["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if path.stat().st_size != record["size"] or digest != record["sha256"]:
        raise SystemExit(f"model artifact mismatch: {path}")
result = {"files": len(manifest["files"]), "resolved_revision": manifest["resolved_revision"], "status": "PASS"}
temporary = Path(str(output) + ".tmp")
with temporary.open("x") as stream:
    json.dump(result, stream, sort_keys=True, indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
temporary.replace(output)
PY

python3 - "$dataset_dir" "$EXPECTED_DATASET_ID" "$qual_dir/dataset-validation.json" <<'PY'
import hashlib, json, os, sys
from pathlib import Path
root = Path(sys.argv[1])
expected_content_id = sys.argv[2]
output = Path(sys.argv[3])
manifest = json.loads((root / "manifest.json").read_text())
expected_files = {
    "train.parquet": {"rows": 4, "sha256": "dda98b10be9a17394459b72ba6d8f096bf8e67bdefb18e25f97da3b9a1e9ceff", "size": 6049},
    "validation.parquet": {"rows": 2, "sha256": "11c361118cb3e605b222555739c2fa1eb06c6d6208b37a3abd4375965b41d982", "size": 5653},
}
if manifest.get("artifact") != "skyrl-multiply-lifecycle-v1" or manifest.get("schema_version") != 1:
    raise SystemExit("dataset artifact/schema mismatch")
if manifest.get("files") != expected_files:
    raise SystemExit("dataset file manifest mismatch")
computed_content_id = hashlib.sha256(json.dumps(expected_files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if manifest.get("content_id") != expected_content_id or computed_content_id != expected_content_id:
    raise SystemExit("dataset content ID mismatch")
if root.name != f"skyrl-multiply-lifecycle-{expected_content_id[:16]}":
    raise SystemExit("dataset directory/content ID mismatch")
if {path.name for path in root.iterdir()} != {"manifest.json", *expected_files}:
    raise SystemExit("dataset contains unexpected or missing entries")
if root.is_symlink() or root.stat().st_mode & 0o222:
    raise SystemExit("dataset directory is symlinked or writable")
manifest_path = root / "manifest.json"
if manifest_path.is_symlink() or manifest_path.stat().st_mode & 0o222:
    raise SystemExit("dataset manifest is symlinked or writable")
for name, record in manifest["files"].items():
    path = root / name
    if path.is_symlink() or not path.is_file() or path.stat().st_mode & 0o222:
        raise SystemExit(f"dataset artifact is symlinked, missing, or writable: {path}")
    if path.stat().st_size != record["size"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
        raise SystemExit(f"dataset artifact mismatch: {path}")
result = {"content_id": manifest["content_id"], "files": manifest["files"], "status": "PASS"}
temporary = Path(str(output) + ".tmp")
with temporary.open("x") as stream:
    json.dump(result, stream, sort_keys=True, indent=2); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
temporary.replace(output)
PY

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

mkdir "$qual_dir/recipe"
install -m 0444 "$WORKTREE/platform/b300/lifecycle/"{build_dataset.py,qualification_entrypoint.py,run_lifecycle.sh,verify_artifacts.py} "$qual_dir/recipe/"
git -C "$WORKTREE" status --short >"$qual_dir/worktree-status.txt"
git -C "$WORKTREE" rev-parse HEAD >"$qual_dir/worktree-revision.txt"
git -C "$WORKTREE" remote -v >"$qual_dir/worktree-remotes.txt"
git -C "$WORKTREE" diff --binary >"$qual_dir/worktree.patch"
docker image inspect "$EXPECTED_IMAGE_ID" >"$qual_dir/image-inspect.json"
cp "$MODEL_DIR/MODEL_MANIFEST.json" "$qual_dir/model-manifest.json"
cp "$dataset_dir/manifest.json" "$qual_dir/dataset-manifest.json"
cp "$payload_dir/PAYLOAD.sha256" "$qual_dir/payload.sha256"

cmd=(
  /opt/venvs/skyrl-megatron/bin/python /opt/qualification/qualification_entrypoint.py
  "data.train_data=[\"$dataset_dir/train.parquet\"]"
  "data.val_data=[\"$dataset_dir/validation.parquet\"]"
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
  trainer.epochs=1
  trainer.max_training_steps=1
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
  trainer.resume_mode=null
  trainer.resume_path=null
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
  generator.inference_engine.num_engines=1
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

docker_args=(
  create
  --pull never
  --name "$container_name"
  --label "io.mercor.qualification.run-id=$run_id"
  --label "io.mercor.qualification.owner=$OWNER_LABEL"
  --init
  --stop-timeout 60
  --gpus "\"device=$gpu_ids\""
  --network host
  --ipc host
  --hostname "$(hostname -s)"
  --user "$(id -u):$(id -g)"
  --ulimit memlock=-1:-1
  --ulimit stack=67108864:67108864
  --ulimit nofile=1048576:1048576
  --mount type=bind,src=/shared,dst=/shared,readonly
  --mount "type=bind,src=$qual_dir,dst=$qual_dir"
  --mount "type=bind,src=$CHECKPOINT_ROOT/$run_id,dst=$CHECKPOINT_ROOT/$run_id"
  --mount "type=bind,src=$scratch_dir,dst=$scratch_dir"
  --mount "type=bind,src=$payload_dir/platform/b300/lifecycle/qualification_entrypoint.py,dst=/opt/qualification/qualification_entrypoint.py,readonly"
  --mount "type=bind,src=$payload_dir/skyrl/backends/skyrl_train/inference_servers/common.py,dst=/workspace/SkyRL/skyrl/backends/skyrl_train/inference_servers/common.py,readonly"
  --mount "type=bind,src=$payload_dir/skyrl/backends/skyrl_train/inference_servers/utils.py,dst=/workspace/SkyRL/skyrl/backends/skyrl_train/inference_servers/utils.py,readonly"
  --mount "type=bind,src=$payload_dir/skyrl/backends/skyrl_train/inference_servers/vllm_router.py,dst=/workspace/SkyRL/skyrl/backends/skyrl_train/inference_servers/vllm_router.py,readonly"
  --mount "type=bind,src=$payload_dir/skyrl/backends/skyrl_train/inference_servers/vllm_server_actor.py,dst=/workspace/SkyRL/skyrl/backends/skyrl_train/inference_servers/vllm_server_actor.py,readonly"
  --mount "type=bind,src=$payload_dir/skyrl/backends/skyrl_train/weight_sync/broadcast_strategy.py,dst=/workspace/SkyRL/skyrl/backends/skyrl_train/weight_sync/broadcast_strategy.py,readonly"
  --mount "type=bind,src=$payload_dir/skyrl/train/utils/utils.py,dst=/workspace/SkyRL/skyrl/train/utils/utils.py,readonly"
  --env "NVIDIA_VISIBLE_DEVICES=$gpu_ids"
  --env "CUDA_VISIBLE_DEVICES=$gpu_ids"
  --env SKYRL_INFERENCE_BIND_HOST=127.0.0.1
  --env SKYRL_INFERENCE_ADVERTISE_HOST=127.0.0.1
  --env SKYRL_WEIGHT_SYNC_MASTER_ADDR=127.0.0.1
  --env "SKYRL_QUAL_RESULT_DIR=$result_dir"
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
  --env "HOME=$scratch_dir/h"
  --env "TMPDIR=$scratch_dir/t"
  --env "RAY_TMPDIR=$scratch_dir/r"
  --env "XDG_CACHE_HOME=$scratch_dir/xdg"
  --env "TRITON_CACHE_DIR=$scratch_dir/triton"
  --env "VLLM_CACHE_ROOT=$scratch_dir/vllm"
)
while IFS= read -r device; do
  docker_args+=(--device "$device:$device")
done < <(find /dev/infiniband -maxdepth 1 -type c -print | sort)
docker_args+=("$EXPECTED_IMAGE_ID" "${cmd[@]}")

printf '%s\0' docker "${docker_args[@]}" | jq -Rs 'split("\u0000")[:-1]' >"$qual_dir/docker-create-command.json"
container_id=$(docker "${docker_args[@]}")
verify_container_identity "$container_id" || die "new container identity mismatch"
docker inspect "$container_id" >"$qual_dir/container-prestart-inspect.json"

set +e
timeout --foreground --signal=TERM --kill-after=60s 1800s docker start --attach "$container_id" 2>&1 | tee "$qual_dir/attached.log"
start_rc=${PIPESTATUS[0]}
set -e
docker inspect "$container_id" >"$qual_dir/container-postexit-inspect.json"
container_exit_code=$(docker inspect "$container_id" --format '{{.State.ExitCode}}')
cleanup_owned_container
snapshot post
: >"$qual_dir/post-selected-gpu-processes.csv"

[[ "$start_rc" -eq 0 ]] || die "bounded container start failed or timed out: rc=$start_rc"
[[ "$container_exit_code" -eq 0 ]] || die "container exit code was $container_exit_code"

for gpu in "$rollout_gpu" "$policy_gpu"; do
  for _ in $(seq 1 30); do
    used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    [[ "$used" == "0" ]] && break
    sleep 2
  done
  [[ "$used" == "0" ]] || die "GPU $gpu still has ${used} MiB allocated after shutdown"
  gpu_uuid=$(nvidia-smi --id="$gpu" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
  if nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader | awk -F, -v uuid="$gpu_uuid" '$1 == uuid {found=1} END {exit !found}'; then
    die "GPU $gpu still has an active compute process after shutdown"
  fi
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader |
    awk -F, -v uuid="$gpu_uuid" '$1 == uuid {print}' >>"$qual_dir/post-selected-gpu-processes.csv"
done

[[ -z $(docker ps -aq --filter "label=io.mercor.qualification.run-id=$run_id") ]] || die "run-owned container remains after shutdown"
ps -eo comm=,args= | awk '
  $1 == "raylet" || $1 == "gcs_server" ||
  $0 ~ /ray::/ || $0 ~ /vllm\.entrypoints/ || $0 ~ /qualification_entrypoint\.py/ {print}
' >"$qual_dir/post-conflicting-processes.txt"
[[ ! -s "$qual_dir/post-conflicting-processes.txt" ]] || die "run-owned Ray/vLLM/SkyRL processes remain after shutdown"

/shared/environments/b300/venvs/skyrl-megatron-0f3a2126/bin/python \
  "$qual_dir/recipe/verify_artifacts.py" \
  --result-dir "$result_dir" \
  --checkpoint-dir "$checkpoint_dir" \
  --export-dir "$export_dir" \
  --attached-log "$qual_dir/attached.log" \
  --post-gpu-processes "$qual_dir/post-selected-gpu-processes.csv" \
  --post-containers "$qual_dir/post-containers.txt" \
  --post-processes "$qual_dir/post-processes.txt" \
  --run-id "$run_id" \
  | tee "$qual_dir/artifact-verification.stdout"

(
  cd "$CHECKPOINT_ROOT/$run_id"
  find . -type f -print0 | sort -z | xargs -0 sha256sum
) >"$qual_dir/checkpoint-files.sha256"
printf '{"container_exit_code":0,"gpu_ids":"%s","run_id":"%s","scratch_preserved":"%s","status":"PASS"}\n' \
  "$gpu_ids" "$run_id" "$scratch_dir" >"$qual_dir/launcher-result.json"
(
  cd "$qual_dir"
  find . -type f ! -name EVIDENCE.sha256 -print0 | sort -z | xargs -0 sha256sum
) >"$qual_dir/EVIDENCE.sha256"

chmod -R a-w "$qual_dir" "$CHECKPOINT_ROOT/$run_id"
finalized=true
trap - EXIT INT TERM
printf 'PASS %s\n' "$run_id"
