#!/usr/bin/env bash
set -Eeuo pipefail

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly WORKTREE="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
readonly IMAGE_REF="skyrl-megatron-b300-cu128-canary:7c528991c4f9-r1"
readonly IMAGE_ID="sha256:e98a7978ad815edbd55d460f0f45ec059dcc2df4584e5c8da9c6183b99b2940c"
readonly QUAL_ROOT="/shared/environments/b300/qualifications"
readonly OWNER_LABEL="skyrl-serving-topology"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

execute=false
while (($#)); do
  case "$1" in
    --execute) execute=true; shift ;;
    -h|--help)
      printf 'Usage: %s --execute\n' "$0"
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "$execute" == true ]] || die "refusing model download without --execute"
[[ $(hostname -s) == ip-172-31-67-119 ]] || die "run from aws-b300-node1"
[[ -z $(git -C "$WORKTREE" status --short) ]] || die "worktree must be clean"
[[ $(docker image inspect "$IMAGE_REF" --format '{{.Id}}') == "$IMAGE_ID" ]] || die "image ID drift"

run_id="$(date -u +%Y%m%dT%H%M%SZ)-stage-qwen-serving-models-r1"
qual_dir="$QUAL_ROOT/$run_id"
[[ ! -e "$qual_dir" ]] || die "qualification path exists: $qual_dir"
mkdir -p "$qual_dir/recipe"

readonly dense_container_name="b300-qwen-stage-dense-${run_id:0:15}"
readonly moe_container_name="b300-qwen-stage-moe-${run_id:0:15}"
container_names=("$dense_container_name" "$moe_container_name")
finalized=false

cleanup() {
  local name
  for name in "${container_names[@]}"; do
    docker container rm --force "$name" >/dev/null 2>&1 || true
  done
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
    cleanup
    printf '{"exit_code":%d,"run_id":"%s","status":"FAIL"}\n' "$rc" "$run_id" >"$qual_dir/launcher-result.json" 2>/dev/null || true
    freeze_evidence || true
    finalized=true
  fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

install -m 0444 "$SCRIPT_DIR/prepare_model_snapshot.py" "$SCRIPT_DIR/stage_qwen_topology_models.sh" "$qual_dir/recipe/"
git -C "$WORKTREE" rev-parse HEAD >"$qual_dir/worktree-revision.txt"
git -C "$WORKTREE" status --short >"$qual_dir/worktree-status.txt"
git -C "$WORKTREE" remote -v >"$qual_dir/worktree-remotes.txt"
docker image inspect "$IMAGE_ID" >"$qual_dir/image-inspect.json"

stage_one() {
  local label=$1
  local name=$2
  local repo_id=$3
  local revision=$4
  local destination=$5
  printf '%s\0' docker run --rm --pull=never --name "$name" \
    --label "io.mercor.qualification.run-id=$run_id" \
    --label "io.mercor.qualification.owner=$OWNER_LABEL" \
    --network host \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,src=/shared/models,dst=/shared/models \
    --mount type=bind,src=/opt/dlami/nvme/cache,dst=/opt/dlami/nvme/cache \
    --mount "type=bind,src=$SCRIPT_DIR/prepare_model_snapshot.py,dst=/opt/prepare_model_snapshot.py,readonly" \
    --env HF_HOME=/opt/dlami/nvme/cache/huggingface \
    --env HF_HUB_CACHE=/opt/dlami/nvme/cache/huggingface/hub \
    --env HF_HUB_DISABLE_TELEMETRY=1 \
    --env DO_NOT_TRACK=1 \
    "$IMAGE_ID" /opt/venvs/skyrl-megatron/bin/python /opt/prepare_model_snapshot.py \
    --repo-id "$repo_id" --revision "$revision" --destination "$destination" \
    --image-ref "$IMAGE_REF" --image-id "$IMAGE_ID" --run-id "$run_id-$label" \
    | jq -Rs 'split("\u0000")[:-1]' >"$qual_dir/command-$label.json"
  docker run --rm --pull=never --name "$name" \
    --label "io.mercor.qualification.run-id=$run_id" \
    --label "io.mercor.qualification.owner=$OWNER_LABEL" \
    --network host \
    --user "$(id -u):$(id -g)" \
    --mount type=bind,src=/shared/models,dst=/shared/models \
    --mount type=bind,src=/opt/dlami/nvme/cache,dst=/opt/dlami/nvme/cache \
    --mount "type=bind,src=$SCRIPT_DIR/prepare_model_snapshot.py,dst=/opt/prepare_model_snapshot.py,readonly" \
    --env HF_HOME=/opt/dlami/nvme/cache/huggingface \
    --env HF_HUB_CACHE=/opt/dlami/nvme/cache/huggingface/hub \
    --env HF_HUB_DISABLE_TELEMETRY=1 \
    --env DO_NOT_TRACK=1 \
    "$IMAGE_ID" /opt/venvs/skyrl-megatron/bin/python /opt/prepare_model_snapshot.py \
    --repo-id "$repo_id" --revision "$revision" --destination "$destination" \
    --image-ref "$IMAGE_REF" --image-id "$IMAGE_ID" --run-id "$run_id-$label" \
    >"$qual_dir/stage-$label.log" 2>&1
}

stage_one dense "$dense_container_name" Qwen/Qwen3.8-27B 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 \
  /shared/models/qwen3.8-27b-1d4bf0f &
dense_pid=$!
stage_one moe "$moe_container_name" Qwen/Qwen3.6-35B-A3B 995ad96eacd98c81ed38be0c5b274b04031597b0 \
  /shared/models/qwen3.6-35b-a3b-995ad96 &
moe_pid=$!
wait "$dense_pid"
wait "$moe_pid"

cp /shared/models/qwen3.8-27b-1d4bf0f/MODEL_MANIFEST.json "$qual_dir/dense-model-manifest.json"
cp /shared/models/qwen3.6-35b-a3b-995ad96/MODEL_MANIFEST.json "$qual_dir/moe-model-manifest.json"
cleanup
printf '{"dense_model":"/shared/models/qwen3.8-27b-1d4bf0f","moe_model":"/shared/models/qwen3.6-35b-a3b-995ad96","run_id":"%s","status":"PASS"}\n' \
  "$run_id" >"$qual_dir/launcher-result.json"
freeze_evidence
finalized=true
trap - EXIT INT TERM
printf 'PASS %s\n' "$run_id"
