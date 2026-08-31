#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly WORKTREE="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
readonly IMAGE_REF="skyrl-megatron-b300-cu128-canary:7c528991c4f9-r1"
readonly IMAGE_ID="sha256:e98a7978ad815edbd55d460f0f45ec059dcc2df4584e5c8da9c6183b99b2940c"
readonly SOURCE_REVISION="7c528991c4f9d470dd9295e10589d99dc3e05053"
readonly LOCK_SHA256="0f3a2126b68747e7d4b854574e9e418c0c4a8f6c9f605865600b04e5d0a2a537"
readonly QUAL_ROOT="/shared/environments/b300/qualifications"
readonly OWNER_LABEL="skyrl-serving-fullnode"
readonly ROUTER_PORT=18080
readonly FIRST_SERVER_PORT=18100

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

usage() {
  printf 'Usage: %s --execute --model {dense|moe} [--phases comma,separated,list]\n' "$0"
}

execute=false
model_family=""
requested_phases=""
while (($#)); do
  case "$1" in
    --execute) execute=true; shift ;;
    --model) model_family=${2:?}; shift 2 ;;
    --phases) requested_phases=${2:?}; shift 2 ;;
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
[[ -z $(docker ps -aq --filter "label=io.mercor.qualification.owner=$OWNER_LABEL") ]] || die "another full-node serving container exists"

if [[ "$model_family" == dense ]]; then
  readonly MODEL_PATH="/shared/models/qwen3.8-27b-1d4bf0f"
  readonly MODEL_REPO="Qwen/Qwen3.8-27B"
  readonly MODEL_REVISION="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
  allowed_phases=(r8tp1 tp8)
else
  readonly MODEL_PATH="/shared/models/qwen3.6-35b-a3b-995ad96"
  readonly MODEL_REPO="Qwen/Qwen3.6-35B-A3B"
  readonly MODEL_REVISION="995ad96eacd98c81ed38be0c5b274b04031597b0"
  allowed_phases=(r8tp1 r2tp4 dp8ep)
fi
if [[ -n "$requested_phases" ]]; then
  IFS=, read -r -a phases <<<"$requested_phases"
else
  phases=("${allowed_phases[@]}")
fi
((${#phases[@]})) || die "at least one phase is required"
for requested_phase in "${phases[@]}"; do
  supported=false
  for allowed_phase in "${allowed_phases[@]}"; do
    [[ "$requested_phase" != "$allowed_phase" ]] || supported=true
  done
  [[ "$supported" == true ]] || die "unsupported $model_family phase: $requested_phase"
done

run_id="$(date -u +%Y%m%dT%H%M%SZ)-vllm-${model_family}-fullnode-b300-r1"
qual_dir="$QUAL_ROOT/$run_id"
result_dir="$qual_dir/results"
[[ ! -e "$qual_dir" ]] || die "qualification path exists: $qual_dir"
mkdir -p "$result_dir" "$qual_dir/recipe" "$qual_dir/phases"

active_phase=""
active_client_name=""
active_router_id=""
active_router_name=""
active_server_ids=()
active_server_names=()
monitor_pid=""
finalized=false

owned_container() {
  local container_id=$1
  [[ $(docker inspect "$container_id" --format '{{index .Config.Labels "io.mercor.qualification.run-id"}}|{{index .Config.Labels "io.mercor.qualification.owner"}}') == "$run_id|$OWNER_LABEL" ]]
}

cleanup_active() {
  local index container_id
  if [[ -n "$active_client_name" ]] && docker container inspect "$active_client_name" >/dev/null 2>&1; then
    docker container rm --force "$active_client_name" >/dev/null 2>&1 || true
  fi
  active_client_name=""
  if [[ -n "$active_router_id" ]] && docker container inspect "$active_router_id" >/dev/null 2>&1; then
    if owned_container "$active_router_id"; then
      docker logs --timestamps "$active_router_id" >"$qual_dir/phases/$active_phase-router-failure.log" 2>&1 || true
      docker container stop --time 30 "$active_router_id" >/dev/null 2>&1 || true
      docker container rm "$active_router_id" >/dev/null 2>&1 || true
    else
      printf 'ERROR: refusing cleanup of mismatched router %s\n' "$active_router_id" >&2
    fi
  fi
  active_router_id=""
  active_router_name=""
  for index in "${!active_server_ids[@]}"; do
    container_id=${active_server_ids[$index]}
    if docker container inspect "$container_id" >/dev/null 2>&1; then
      if owned_container "$container_id"; then
        docker logs --timestamps "$container_id" >"$qual_dir/phases/$active_phase-server-$index-failure.log" 2>&1 || true
        docker container stop --time 60 "$container_id" >/dev/null 2>&1 || true
        docker container rm "$container_id" >/dev/null 2>&1 || true
      else
        printf 'ERROR: refusing cleanup of mismatched server %s\n' "$container_id" >&2
      fi
    fi
  done
  active_server_ids=()
  active_server_names=()
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" >/dev/null 2>&1 || true
  fi
  monitor_pid=""
}

capture_snapshot() {
  local label=$1
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits >"$qual_dir/$label-gpus.csv"
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

wait_for_gpu_cleanup() {
  local gpu used clean
  for _ in $(seq 1 90); do
    clean=true
    for gpu in {0..7}; do
      used=$(nvidia-smi --id="$gpu" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
      [[ "$used" == 0 ]] || clean=false
    done
    [[ "$clean" == true ]] && return 0
    sleep 2
  done
  return 1
}

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
  "$SCRIPT_DIR/"{prepare_model_snapshot.py,run_vllm_fullnode_comparison.sh,summarize_fullnode_results.py,vllm_bench_serve.py} \
  "$qual_dir/recipe/"
cp "$MODEL_PATH/MODEL_MANIFEST.json" "$qual_dir/model-manifest.json"
git -C "$WORKTREE" rev-parse HEAD >"$qual_dir/worktree-revision.txt"
git -C "$WORKTREE" status --short >"$qual_dir/worktree-status.txt"
git -C "$WORKTREE" remote -v >"$qual_dir/worktree-remotes.txt"
docker image inspect "$IMAGE_ID" >"$qual_dir/image-inspect.json"
printf '%s\n' "${phases[@]}" >"$qual_dir/selected-phases.txt"
docker run --rm --pull=never --network none --entrypoint /opt/venvs/skyrl-megatron/bin/python "$IMAGE_ID" \
  -c 'import importlib.metadata as m, json; print(json.dumps({p: m.version(p) for p in ("torch", "transformers", "vllm", "vllm-router")}, sort_keys=True))' \
  >"$qual_dir/runtime-versions.json"

SERVER_GPU_SPECS=()
SERVER_TPS=()
SERVER_PORTS=()
SERVER_EXTRAS=()

set_phase_layout() {
  local phase=$1 gpu
  SERVER_GPU_SPECS=()
  SERVER_TPS=()
  SERVER_PORTS=()
  SERVER_EXTRAS=()
  case "$phase" in
    r8tp1)
      for gpu in {0..7}; do
        SERVER_GPU_SPECS+=("$gpu")
        SERVER_TPS+=("1")
        SERVER_PORTS+=("$((FIRST_SERVER_PORT + gpu))")
        SERVER_EXTRAS+=("")
      done
      ;;
    tp8)
      SERVER_GPU_SPECS+=("0,1,2,3,4,5,6,7")
      SERVER_TPS+=("8")
      SERVER_PORTS+=("$FIRST_SERVER_PORT")
      SERVER_EXTRAS+=("")
      ;;
    r2tp4)
      SERVER_GPU_SPECS+=("0,1,2,3" "4,5,6,7")
      SERVER_TPS+=("4" "4")
      SERVER_PORTS+=("$FIRST_SERVER_PORT" "$((FIRST_SERVER_PORT + 1))")
      SERVER_EXTRAS+=("" "")
      ;;
    dp8ep)
      SERVER_GPU_SPECS+=("0,1,2,3,4,5,6,7")
      SERVER_TPS+=("1")
      SERVER_PORTS+=("$FIRST_SERVER_PORT")
      SERVER_EXTRAS+=("--data-parallel-size 8 --data-parallel-size-local 8 --enable-expert-parallel")
      ;;
    *) die "unsupported phase layout: $phase" ;;
  esac
}

create_server() {
  local phase=$1 index=$2 gpu_spec=$3 tp=$4 port=$5 extra_text=$6
  local scratch="/opt/dlami/nvme/cache/tmp/vf/${run_id:0:15}-${phase}-${index}"
  local name="b300-vllm-${run_id:0:15}-${phase}-${index}"
  local -a extra=() model_kernel_args=()
  [[ -z "$extra_text" ]] || read -r -a extra <<<"$extra_text"
  [[ "$model_family" != moe ]] || model_kernel_args=(--moe-backend triton)
  [[ ! -e "$scratch" ]] || die "scratch path exists: $scratch"
  mkdir -p "$scratch"/{home,tmp,xdg,triton,vllm}
  local -a command=(
    docker create --pull=never --name "$name"
    --label "io.mercor.qualification.run-id=$run_id"
    --label "io.mercor.qualification.owner=$OWNER_LABEL"
    --label "io.mercor.qualification.phase=$phase"
    --label "io.mercor.qualification.replica=$index"
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
    --port "$port"
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
    "${model_kernel_args[@]}"
    --compilation-config '{"pass_config":{"fuse_allreduce_rms":false}}'
    --no-enable-log-requests
    "${extra[@]}"
  )
  printf '%s\0' "${command[@]}" | jq -Rs 'split("\u0000")[:-1]' >"$qual_dir/phases/$phase-server-$index-command.json"
  local container_id
  container_id=$("${command[@]}")
  active_server_ids+=("$container_id")
  active_server_names+=("$name")
  docker inspect "$container_id" >"$qual_dir/phases/$phase-server-$index-container-prestart.json"
  jq -n --arg scratch "$scratch" --arg gpu_spec "$gpu_spec" --argjson index "$index" --argjson port "$port" --argjson tp "$tp" \
    '{gpu_spec:$gpu_spec,index:$index,port:$port,scratch_preserved:$scratch,tensor_parallel_size:$tp}' \
    >"$qual_dir/phases/$phase-server-$index-layout.json"
}

start_router() {
  local phase=$1
  shift
  local -a worker_urls=("$@")
  local scratch="/opt/dlami/nvme/cache/tmp/vf/${run_id:0:15}-${phase}-router"
  active_router_name="b300-vllm-router-${run_id:0:15}-${phase}"
  [[ ! -e "$scratch" ]] || die "router scratch path exists: $scratch"
  mkdir -p "$scratch/home"
  local -a command=(
    docker create --pull=never --name "$active_router_name"
    --label "io.mercor.qualification.run-id=$run_id"
    --label "io.mercor.qualification.owner=$OWNER_LABEL"
    --label "io.mercor.qualification.phase=$phase"
    --init --stop-timeout 30
    --network host
    --user "$(id -u):$(id -g)"
    --mount type=bind,src=/shared,dst=/shared,readonly
    --mount "type=bind,src=$scratch,dst=/c"
    --env HOME=/c/home
    --env PYTHONDONTWRITEBYTECODE=1
    --env PYTHONNOUSERSITE=1
    "$IMAGE_ID"
    /opt/venvs/skyrl-megatron/bin/vllm-router
    --host 127.0.0.1
    --port "$ROUTER_PORT"
    --worker-urls "${worker_urls[@]}"
    --policy round_robin
    --worker-startup-timeout-secs 300
    --worker-startup-check-interval 1
    --request-timeout-secs 900
    --max-concurrent-requests 1024
    --queue-size 2048
    --queue-timeout-secs 900
    --log-level info
  )
  printf '%s\0' "${command[@]}" | jq -Rs 'split("\u0000")[:-1]' >"$qual_dir/phases/$phase-router-command.json"
  active_router_id=$("${command[@]}")
  docker inspect "$active_router_id" >"$qual_dir/phases/$phase-router-container-prestart.json"
  docker start "$active_router_id" >"$qual_dir/phases/$phase-router-container-start.txt"
  for _ in $(seq 1 120); do
    if curl -fsS --max-time 5 "http://127.0.0.1:$ROUTER_PORT/health" >"$qual_dir/phases/$phase-router-health.json.tmp" 2>/dev/null; then
      mv "$qual_dir/phases/$phase-router-health.json.tmp" "$qual_dir/phases/$phase-router-health.json"
      return 0
    fi
    [[ $(docker inspect "$active_router_id" --format '{{.State.Running}}') == true ]] || break
    sleep 1
  done
  docker logs --timestamps "$active_router_id" >"$qual_dir/phases/$phase-router.log" 2>&1 || true
  die "router failed readiness for $phase"
}

run_benchmark_cell() {
  local phase=$1 concurrency=$2
  local prompts=$((concurrency * 2))
  local cell="${phase}-i1024-o256-c${concurrency}"
  active_client_name="b300-vllm-client-${run_id:0:15}-${phase}-${concurrency}"
  local -a command=(
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
    --base-url "http://127.0.0.1:$ROUTER_PORT"
    --endpoint /v1/completions
    --model b300-qwen
    --tokenizer "$MODEL_PATH"
    --dataset-name random
    --random-input-len 1024
    --random-output-len 256
    --random-range-ratio 0
    --num-prompts "$prompts"
    --request-rate inf
    --max-concurrency "$concurrency"
    --num-warmups 16
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
    --metadata "model_family=$model_family" "topology=$phase" "gpu_count=8" \
      "input_tokens=1024" "output_tokens=256"
  )
  printf '%s\0' "${command[@]}" | jq -Rs 'split("\u0000")[:-1]' >"$qual_dir/phases/$cell-command.json"
  timeout --foreground --signal=TERM --kill-after=30s 1800s "${command[@]}" >"$qual_dir/phases/$cell.log" 2>&1
  active_client_name=""
  jq -e '.completed == .num_prompts and .failed == 0' "$result_dir/$cell.json" >/dev/null || die "benchmark cell incomplete: $cell"
}

finish_phase() {
  local phase=$1 index container_id count sum=0
  docker stop --time 30 "$active_router_id" >"$qual_dir/phases/$phase-router-container-stop.txt"
  docker inspect "$active_router_id" >"$qual_dir/phases/$phase-router-container-final.json"
  docker logs --timestamps "$active_router_id" >"$qual_dir/phases/$phase-router.log" 2>&1 || true
  docker rm "$active_router_id" >"$qual_dir/phases/$phase-router-container-remove.txt"
  active_router_id=""
  active_router_name=""
  : >"$qual_dir/phases/$phase-request-distribution.tsv"
  for index in "${!active_server_ids[@]}"; do
    container_id=${active_server_ids[$index]}
    docker stop --time 60 "$container_id" >"$qual_dir/phases/$phase-server-$index-container-stop.txt"
    docker inspect "$container_id" >"$qual_dir/phases/$phase-server-$index-container-final.json"
    docker logs --timestamps "$container_id" >"$qual_dir/phases/$phase-server-$index.log" 2>&1 || true
    count=$(rg -c 'POST /v1/completions HTTP/1.1.*200 OK' "$qual_dir/phases/$phase-server-$index.log" || true)
    printf '%s\t%s\n' "$index" "$count" >>"$qual_dir/phases/$phase-request-distribution.tsv"
    ((sum += count))
    ((count > 0)) || die "server $index in $phase served no benchmark requests"
    docker rm "$container_id" >"$qual_dir/phases/$phase-server-$index-container-remove.txt"
  done
  active_server_ids=()
  active_server_names=()
  ((sum == 1312)) || die "unexpected successful request count for $phase: $sum"
  jq -Rn \
    '[inputs | split("\t") | {replica:(.[0] | tonumber),successful_requests:(.[1] | tonumber)}] | {replicas:.,successful_requests:(map(.successful_requests) | add)}' \
    <"$qual_dir/phases/$phase-request-distribution.tsv" >"$qual_dir/phases/$phase-request-distribution.json"
  if [[ -n "$monitor_pid" ]]; then
    kill "$monitor_pid" >/dev/null 2>&1 || true
    wait "$monitor_pid" >/dev/null 2>&1 || true
    monitor_pid=""
  fi
  wait_for_gpu_cleanup || die "GPUs did not clean up after $phase"
  jq -n --arg phase "$phase" --argjson replicas "${#SERVER_GPU_SPECS[@]}" --argjson successful_requests "$sum" \
    '{phase:$phase,replicas:$replicas,status:"PASS",successful_requests:$successful_requests}' \
    >"$qual_dir/phases/$phase-result.json"
  active_phase=""
}

run_phase() {
  local phase=$1 index all_ready failed
  local -a worker_urls=()
  active_phase=$phase
  active_server_ids=()
  active_server_names=()
  wait_for_gpu_cleanup || die "GPUs did not begin clean for $phase"
  set_phase_layout "$phase"
  for index in "${!SERVER_GPU_SPECS[@]}"; do
    create_server "$phase" "$index" "${SERVER_GPU_SPECS[$index]}" "${SERVER_TPS[$index]}" \
      "${SERVER_PORTS[$index]}" "${SERVER_EXTRAS[$index]}"
    worker_urls+=("http://127.0.0.1:${SERVER_PORTS[$index]}")
  done
  date -u +%s.%N >"$qual_dir/phases/$phase-start-epoch.txt"
  for index in "${!active_server_ids[@]}"; do
    docker start "${active_server_ids[$index]}" >"$qual_dir/phases/$phase-server-$index-container-start.txt"
  done
  timeout --signal=TERM 3600s nvidia-smi dmon -i 0,1,2,3,4,5,6,7 -s pucm -d 1 >"$qual_dir/phases/$phase-gpu-dmon.log" 2>&1 &
  monitor_pid=$!

  for _ in $(seq 1 240); do
    all_ready=true
    failed=false
    for index in "${!active_server_ids[@]}"; do
      if [[ -s "$qual_dir/phases/$phase-server-$index-models.json" ]]; then
        continue
      fi
      if curl -fsS --max-time 5 "${worker_urls[$index]}/v1/models" >"$qual_dir/phases/$phase-server-$index-models.json.tmp" 2>/dev/null; then
        mv "$qual_dir/phases/$phase-server-$index-models.json.tmp" "$qual_dir/phases/$phase-server-$index-models.json"
      else
        all_ready=false
        [[ $(docker inspect "${active_server_ids[$index]}" --format '{{.State.Running}}') == true ]] || failed=true
      fi
    done
    [[ "$all_ready" != true ]] || break
    [[ "$failed" != true ]] || break
    sleep 5
  done
  date -u +%s.%N >"$qual_dir/phases/$phase-ready-epoch.txt"
  for index in "${!active_server_ids[@]}"; do
    if [[ ! -s "$qual_dir/phases/$phase-server-$index-models.json" ]]; then
      docker logs --timestamps "${active_server_ids[$index]}" >"$qual_dir/phases/$phase-server-$index.log" 2>&1 || true
      die "server $index failed readiness for $phase"
    fi
  done
  nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits >"$qual_dir/phases/$phase-loaded-gpus.csv"
  start_router "$phase" "${worker_urls[@]}"
  run_benchmark_cell "$phase" 128
  run_benchmark_cell "$phase" 512
  finish_phase "$phase"
}

for phase in "${phases[@]}"; do
  run_phase "$phase"
done

expected_phases=$(IFS=,; printf '%s' "${phases[*]}")
/shared/environments/b300/venvs/skyrl-megatron-0f3a2126/bin/python "$SCRIPT_DIR/summarize_fullnode_results.py" \
  --result-dir "$result_dir" --model-family "$model_family" --expected-phases "$expected_phases" \
  --output "$qual_dir/fullnode-summary.json" >"$qual_dir/fullnode-summary.stdout"
capture_snapshot post
[[ ! -s "$qual_dir/post-gpu-processes.csv" ]] || die "GPU processes remain after comparison"
[[ -z $(docker ps -aq --filter "label=io.mercor.qualification.run-id=$run_id") ]] || die "run-owned containers remain"
jq -e 'all(.[]; (.[5] | tonumber) == 0)' < <(jq -Rsc 'split("\n")[:-1] | map(split(", "))' "$qual_dir/post-gpus.csv") >/dev/null || die "volatile uncorrected ECC error detected"
printf '{"model_family":"%s","run_id":"%s","status":"PASS"}\n' "$model_family" "$run_id" >"$qual_dir/launcher-result.json"
freeze_evidence
finalized=true
trap - EXIT INT TERM
printf 'PASS %s\n' "$run_id"
