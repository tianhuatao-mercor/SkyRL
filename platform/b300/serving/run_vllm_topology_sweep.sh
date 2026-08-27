#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly WORKTREE="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
readonly IMAGE_REF="skyrl-megatron-b300-cu128-canary:7c528991c4f9-r1"
readonly IMAGE_ID="sha256:e98a7978ad815edbd55d460f0f45ec059dcc2df4584e5c8da9c6183b99b2940c"
readonly SOURCE_REVISION="7c528991c4f9d470dd9295e10589d99dc3e05053"
readonly LOCK_SHA256="0f3a2126b68747e7d4b854574e9e418c0c4a8f6c9f605865600b04e5d0a2a537"
readonly QUAL_ROOT="/shared/environments/b300/qualifications"
readonly OWNER_LABEL="skyrl-serving-topology"
readonly SERVER_PORT=18080

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  printf 'Usage: %s --execute --model {dense|moe}\n' "$0"
}

execute=false
model_family=""
while (($#)); do
  case "$1" in
    --execute) execute=true; shift ;;
    --model) model_family=${2:?}; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$execute" == true ]] || die "refusing GPU benchmark without --execute"
[[ "$model_family" == dense || "$model_family" == moe ]] || die "--model must be dense or moe"
[[ $(hostname -s) == ip-172-31-67-119 ]] || die "run from aws-b300-node1"
[[ -z $(git -C "$WORKTREE" status --short) ]] || die "worktree must be clean"
[[ $(docker image inspect "$IMAGE_REF" --format '{{.Id}}') == "$IMAGE_ID" ]] || die "image ID drift"
[[ $(docker image inspect "$IMAGE_ID" --format '{{index .Config.Labels "org.opencontainers.image.revision"}}') == "$SOURCE_REVISION" ]] || die "image source revision drift"
[[ $(docker image inspect "$IMAGE_ID" --format '{{index .Config.Labels "io.mercor.skyrl.lock-sha256"}}') == "$LOCK_SHA256" ]] || die "image lock drift"
[[ -z $(docker ps -aq --filter "label=io.mercor.qualification.owner=$OWNER_LABEL") ]] || die "another serving-topology container exists"

if [[ "$model_family" == dense ]]; then
  readonly MODEL_PATH="/shared/models/qwen3.8-27b-1d4bf0f"
  readonly MODEL_REPO="Qwen/Qwen3.8-27B"
  readonly MODEL_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
  shapes=(tp1 tp2 tp4 tp8)
else
  readonly MODEL_PATH="/shared/models/qwen3.6-35b-a3b-995ad96"
  readonly MODEL_REPO="Qwen/Qwen3.6-35B-A3B"
  readonly MODEL_REVISION="995ad96eacd98c81ed38be0c5b274b04031597b0"
  shapes=(tp1 tp2 tp2ep tp4 tp4ep dp8ep)
fi

run_id="$(date -u +%Y%m%dT%H%M%SZ)-vllm-${model_family}-topology-b300-r1"
qual_dir="$QUAL_ROOT/$run_id"
result_dir="$qual_dir/results"
[[ ! -e "$qual_dir" ]] || die "qualification path exists: $qual_dir"
mkdir -p "$result_dir" "$qual_dir/recipe" "$qual_dir/shapes"

active_server_id=""
active_server_name=""
active_client_name=""
monitor_pid=""
finalized=false

cleanup_active() {
  if [[ -n "$active_client_name" ]] && docker container inspect "$active_client_name" >/dev/null 2>&1; then
    docker container rm --force "$active_client_name" >/dev/null 2>&1 || true
  fi
  active_client_name=""
  if [[ -n "$active_server_id" ]] && docker container inspect "$active_server_id" >/dev/null 2>&1; then
    if [[ $(docker inspect "$active_server_id" --format '{{index .Config.Labels "io.mercor.qualification.run-id"}}|{{index .Config.Labels "io.mercor.qualification.owner"}}') == "$run_id|$OWNER_LABEL" ]]; then
      docker container stop --time 60 "$active_server_id" >/dev/null 2>&1 || true
      docker container rm "$active_server_id" >/dev/null 2>&1 || true
    else
      printf 'ERROR: refusing cleanup of mismatched container %s\n' "$active_server_id" >&2
    fi
  fi
  active_server_id=""
  active_server_name=""
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" >/dev/null 2>&1 || true
  fi
  monitor_pid=""
}

capture_snapshot() {
  local label=$1
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv,noheader,nounits >"$qual_dir/$label-gpus.csv"
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory --format=csv,noheader >"$qual_dir/$label-gpu-processes.csv" || true
  docker ps -a --no-trunc >"$qual_dir/$label-containers.txt"
  ps -eo comm=,user=,pid=,ppid=,lstart=,args= --sort=pid >"$qual_dir/$label-processes.txt"
  findmnt -T /shared -o TARGET,SOURCE,FSTYPE,OPTIONS >"$qual_dir/$label-fsx.txt"
  date -u +%Y-%m-%dT%H:%M:%S.%NZ >"$qual_dir/$label-timestamp.txt"
}

freeze_evidence() {
  (
    cd "$qual_dir"
    find . -type f ! -name EVIDENCE.sha256 -print0 | sort -z | xargs -0 -r sha256sum >EVIDENCE.sha256
  )
  chmod -R a-w "$qual_dir"
}

on_exit() {
  local rc=$?
  if [[ "$finalized" != true ]]; then
    cleanup_active
    capture_snapshot post-failure || true
    printf '{"exit_code":%d,"model_family":"%s","run_id":"%s","status":"FAIL"}\n' \
      "$rc" "$model_family" "$run_id" >"$qual_dir/launcher-result.json" 2>/dev/null || true
    freeze_evidence || true
    finalized=true
  fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

/shared/verify-handoff >"$qual_dir/verify-handoff.txt"
capture_snapshot pre
jq -Rse \
  'split("\n") | map(select(length > 0) | split(", ")) | all(.[]; (.[3] | tonumber) == 0)' \
  "$qual_dir/pre-gpus.csv" >/dev/null || die "one or more GPUs are not idle"
[[ ! -s "$qual_dir/pre-gpu-processes.csv" ]] || die "GPU processes already exist"

/shared/environments/b300/venvs/skyrl-megatron-0f3a2126/bin/python "$SCRIPT_DIR/prepare_model_snapshot.py" \
  --repo-id "$MODEL_REPO" --revision "$MODEL_REVISION" --destination "$MODEL_PATH" \
  --image-ref "$IMAGE_REF" --image-id "$IMAGE_ID" --run-id "$run_id-validation" \
  >"$qual_dir/model-validation.json"

install -m 0444 \
  "$SCRIPT_DIR/"{prepare_model_snapshot.py,run_vllm_topology_sweep.sh,summarize_topology_results.py,vllm_bench_serve.py} \
  "$qual_dir/recipe/"
cp "$MODEL_PATH/MODEL_MANIFEST.json" "$qual_dir/model-manifest.json"
git -C "$WORKTREE" rev-parse HEAD >"$qual_dir/worktree-revision.txt"
git -C "$WORKTREE" status --short >"$qual_dir/worktree-status.txt"
git -C "$WORKTREE" remote -v >"$qual_dir/worktree-remotes.txt"
docker image inspect "$IMAGE_ID" >"$qual_dir/image-inspect.json"
docker run --rm --pull=never --network none --entrypoint /opt/venvs/skyrl-megatron/bin/python "$IMAGE_ID" \
  -c 'import importlib.metadata as m, json; print(json.dumps({p: m.version(p) for p in ("torch", "transformers", "vllm")}, sort_keys=True))' \
  >"$qual_dir/runtime-versions.json"

shape_params() {
  local shape=$1
  case "$shape" in
    tp1) printf '0|1|' ;;
    tp2) printf '0,1|2|' ;;
    tp2ep) printf '0,1|2|--enable-expert-parallel' ;;
    tp4) printf '0,1,2,3|4|' ;;
    tp4ep) printf '0,1,2,3|4|--enable-expert-parallel' ;;
    tp8) printf '0,1,2,3,4,5,6,7|8|' ;;
    dp8ep) printf '0,1,2,3,4,5,6,7|1|--data-parallel-size 8 --data-parallel-size-local 8 --enable-expert-parallel' ;;
    *) return 1 ;;
  esac
}

wait_for_gpu_cleanup() {
  local gpu_spec=$1
  local gpu
  IFS=, read -r -a selected <<<"$gpu_spec"
  for _ in $(seq 1 60); do
    local clean=true
    for gpu in "${selected[@]}"; do
      used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
      [[ "$used" == 0 ]] || clean=false
    done
    [[ "$clean" == true ]] && return 0
    sleep 2
  done
  return 1
}

run_benchmark_cell() {
  local shape=$1
  local gpu_count=$2
  local input_tokens=$3
  local output_tokens=$4
  local concurrency=$5
  local prompts=$6
  local cell="${shape}-i${input_tokens}-o${output_tokens}-c${concurrency}"
  active_client_name="b300-vllm-client-${run_id:0:15}-${shape}-${concurrency}"
  client_cmd=(
    docker run --rm --pull=never --name "$active_client_name"
    --label "io.mercor.qualification.run-id=$run_id"
    --label "io.mercor.qualification.owner=$OWNER_LABEL"
    --network host
    --user "$(id -u):$(id -g)"
    --mount type=bind,src=/shared,dst=/shared,readonly
    --mount "type=bind,src=$qual_dir,dst=$qual_dir"
    --env HF_HUB_OFFLINE=1
    --env TRANSFORMERS_OFFLINE=1
    --env HF_DATASETS_OFFLINE=1
    --env HF_HUB_DISABLE_TELEMETRY=1
    --env VLLM_NO_USAGE_STATS=1
    --env PYTHONDONTWRITEBYTECODE=1
    "$IMAGE_ID"
    /opt/venvs/skyrl-megatron/bin/python "$qual_dir/recipe/vllm_bench_serve.py"
    --backend openai
    --base-url "http://127.0.0.1:$SERVER_PORT"
    --endpoint /v1/completions
    --model b300-qwen
    --tokenizer "$MODEL_PATH"
    --dataset-name random
    --random-input-len "$input_tokens"
    --random-output-len "$output_tokens"
    --random-range-ratio 0
    --num-prompts "$prompts"
    --request-rate inf
    --max-concurrency "$concurrency"
    --num-warmups 4
    --temperature 0
    --ignore-eos
    --seed 42
    --disable-tqdm
    --metric-percentiles 50,90,99
    --percentile-metrics ttft,tpot,itl
    --save-result
    --save-detailed
    --result-dir "$result_dir"
    --result-filename "$cell.json"
    --metadata "model_family=$model_family" "topology=$shape" "gpu_count=$gpu_count" \
      "input_tokens=$input_tokens" "output_tokens=$output_tokens"
  )
  printf '%s\0' "${client_cmd[@]}" | jq -Rs 'split("\u0000")[:-1]' >"$qual_dir/shapes/$cell-command.json"
  timeout --foreground --signal=TERM --kill-after=30s 900s "${client_cmd[@]}" >"$qual_dir/shapes/$cell.log" 2>&1
  active_client_name=""
  jq -e '.completed == .num_prompts and .failed == 0' "$result_dir/$cell.json" >/dev/null || die "benchmark cell incomplete: $cell"
}

run_shape() {
  local shape=$1
  local packed gpu_spec tp extra_text
  packed=$(shape_params "$shape") || die "unsupported shape: $shape"
  IFS='|' read -r gpu_spec tp extra_text <<<"$packed"
  local -a extra=()
  [[ -z "$extra_text" ]] || read -r -a extra <<<"$extra_text"
  local gpu_count
  gpu_count=$(awk -F, '{print NF}' <<<"$gpu_spec")
  wait_for_gpu_cleanup "$gpu_spec" || die "selected GPUs did not begin clean for $shape"

  local scratch="/opt/dlami/nvme/cache/tmp/vb/${run_id:0:15}-$shape"
  [[ ! -e "$scratch" ]] || die "scratch path exists: $scratch"
  mkdir -p "$scratch"/{home,tmp,xdg,triton,vllm}
  active_server_name="b300-vllm-${run_id:0:15}-$shape"
  server_cmd=(
    docker create --pull=never --name "$active_server_name"
    --label "io.mercor.qualification.run-id=$run_id"
    --label "io.mercor.qualification.owner=$OWNER_LABEL"
    --label "io.mercor.qualification.topology=$shape"
    --init --stop-timeout 60
    --gpus "\"device=$gpu_spec\""
    --network host --ipc host
    --user "$(id -u):$(id -g)"
    --ulimit memlock=-1:-1
    --ulimit stack=67108864:67108864
    --ulimit nofile=1048576:1048576
    --mount type=bind,src=/shared,dst=/shared,readonly
    --mount "type=bind,src=$scratch,dst=/c"
    --env "NVIDIA_VISIBLE_DEVICES=$gpu_spec"
    --env HF_HUB_OFFLINE=1
    --env TRANSFORMERS_OFFLINE=1
    --env HF_DATASETS_OFFLINE=1
    --env HF_HUB_DISABLE_TELEMETRY=1
    --env VLLM_NO_USAGE_STATS=1
    --env VLLM_USE_FLASHINFER_SAMPLER=0
    --env RAY_USAGE_STATS_ENABLED=0
    --env PYTHONDONTWRITEBYTECODE=1
    --env PYTHONNOUSERSITE=1
    --env NCCL_CUMEM_ENABLE=0
    --env HOME=/c/home
    --env TMPDIR=/c/tmp
    --env XDG_CACHE_HOME=/c/xdg
    --env TRITON_CACHE_DIR=/c/triton
    --env VLLM_CACHE_ROOT=/c/vllm
    "$IMAGE_ID"
    /opt/venvs/skyrl-megatron/bin/vllm serve "$MODEL_PATH"
    --host 127.0.0.1
    --port "$SERVER_PORT"
    --served-model-name b300-qwen
    --language-model-only
    --dtype bfloat16
    --load-format safetensors
    --generation-config vllm
    --tensor-parallel-size "$tp"
    --distributed-executor-backend mp
    --gpu-memory-utilization 0.90
    --max-model-len 32768
    --max-num-seqs 512
    --max-num-batched-tokens 32768
    --enable-chunked-prefill
    --no-enable-prefix-caching
    --attention-backend FLASHINFER
    --no-enable-log-requests
    "${extra[@]}"
  )
  printf '%s\0' "${server_cmd[@]}" | jq -Rs 'split("\u0000")[:-1]' >"$qual_dir/shapes/$shape-server-command.json"
  active_server_id=$("${server_cmd[@]}")
  docker inspect "$active_server_id" >"$qual_dir/shapes/$shape-container-prestart.json"
  date -u +%s.%N >"$qual_dir/shapes/$shape-start-epoch.txt"
  docker start "$active_server_id" >"$qual_dir/shapes/$shape-container-start.txt"
  timeout --signal=TERM 2700s nvidia-smi dmon -i "$gpu_spec" -s pucm -d 1 >"$qual_dir/shapes/$shape-gpu-dmon.log" 2>&1 &
  monitor_pid=$!

  local ready=false
  for _ in $(seq 1 180); do
    if curl -fsS --max-time 5 "http://127.0.0.1:$SERVER_PORT/v1/models" >"$qual_dir/shapes/$shape-models.json.tmp" 2>/dev/null; then
      mv "$qual_dir/shapes/$shape-models.json.tmp" "$qual_dir/shapes/$shape-models.json"
      ready=true
      break
    fi
    [[ $(docker inspect "$active_server_id" --format '{{.State.Running}}') == true ]] || break
    sleep 5
  done
  date -u +%s.%N >"$qual_dir/shapes/$shape-ready-epoch.txt"
  if [[ "$ready" != true ]]; then
    docker logs --timestamps "$active_server_id" >"$qual_dir/shapes/$shape-server.log" 2>&1 || true
    die "server failed readiness for $shape"
  fi
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits >"$qual_dir/shapes/$shape-loaded-gpus.csv"

  run_benchmark_cell "$shape" "$gpu_count" 1024 256 1 4
  run_benchmark_cell "$shape" "$gpu_count" 1024 256 32 64
  run_benchmark_cell "$shape" "$gpu_count" 1024 256 128 256
  run_benchmark_cell "$shape" "$gpu_count" 16384 128 8 16

  docker stop --time 60 "$active_server_id" >"$qual_dir/shapes/$shape-container-stop.txt"
  docker inspect "$active_server_id" >"$qual_dir/shapes/$shape-container-final.json"
  docker logs --timestamps "$active_server_id" >"$qual_dir/shapes/$shape-server.log" 2>&1 || true
  docker rm "$active_server_id" >"$qual_dir/shapes/$shape-container-remove.txt"
  active_server_id=""
  active_server_name=""
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" >/dev/null 2>&1 || true
    monitor_pid=""
  fi
  wait_for_gpu_cleanup "$gpu_spec" || die "selected GPUs did not clean up after $shape"
  jq -n --arg shape "$shape" --arg scratch "$scratch" --argjson gpu_count "$gpu_count" \
    '{gpu_count:$gpu_count,scratch_preserved:$scratch,status:"PASS",topology:$shape}' \
    >"$qual_dir/shapes/$shape-result.json"
}

for shape in "${shapes[@]}"; do
  run_shape "$shape"
done

/shared/environments/b300/venvs/skyrl-megatron-0f3a2126/bin/python "$SCRIPT_DIR/summarize_topology_results.py" \
  --result-dir "$result_dir" --model-family "$model_family" --output "$qual_dir/topology-summary.json" \
  >"$qual_dir/topology-summary.stdout"
capture_snapshot post
[[ ! -s "$qual_dir/post-gpu-processes.csv" ]] || die "GPU processes remain after sweep"
[[ -z $(docker ps -aq --filter "label=io.mercor.qualification.run-id=$run_id") ]] || die "run-owned containers remain"
printf '{"model_family":"%s","run_id":"%s","status":"PASS"}\n' "$model_family" "$run_id" >"$qual_dir/launcher-result.json"
freeze_evidence
finalized=true
trap - EXIT INT TERM
printf 'PASS %s\n' "$run_id"
