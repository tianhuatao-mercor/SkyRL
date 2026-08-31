#!/usr/bin/env bash
set -Eeuo pipefail

readonly WORKTREE="/shared/code/SkyRL-b300-lifecycle"
readonly QUAL_ROOT="/shared/environments/b300/qualifications"
readonly CHECKPOINT_ROOT="/shared/checkpoints/qualifications"
readonly SOURCE_RUN_ID="20260827T000439Z-skyrl-lifecycle-nccl-dense-2eng-r1"
readonly RESUME_RUN_ID="20260827T002835Z-skyrl-lifecycle-resume-2eng-r1"
readonly SOURCE_RUN="$QUAL_ROOT/$SOURCE_RUN_ID"
readonly SOURCE_CHECKPOINT="$CHECKPOINT_ROOT/$SOURCE_RUN_ID"
readonly RESUME_RUN="$QUAL_ROOT/$RESUME_RUN_ID"
readonly RESUME_CHECKPOINT="$CHECKPOINT_ROOT/$RESUME_RUN_ID"

die() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

[[ ${1:-} == --execute && $# -eq 1 ]] || die "usage: $0 --execute"
[[ -z $(git -C "$WORKTREE" status --short) ]] || die "lifecycle worktree must be clean"
for root in "$SOURCE_RUN" "$SOURCE_CHECKPOINT" "$RESUME_RUN" "$RESUME_CHECKPOINT"; do
  [[ -d "$root" ]] || die "required source tree missing: $root"
  [[ ! -w "$root" ]] || die "required source tree must be read-only: $root"
done

run_id="$(date -u +%Y%m%dT%H%M%SZ)-skyrl-lifecycle-resume-audit-r1"
audit_dir="$QUAL_ROOT/$run_id"
[[ ! -e "$audit_dir" ]] || die "refusing to reuse audit path: $audit_dir"
mkdir "$audit_dir"

(
  cd "$SOURCE_RUN"
  sha256sum -c EVIDENCE.sha256
) >"$audit_dir/source-evidence-pre.txt"
(
  cd "$SOURCE_CHECKPOINT"
  sha256sum -c "$SOURCE_RUN/checkpoint-files.sha256"
) >"$audit_dir/source-checkpoint-pre.txt"
(
  cd "$RESUME_RUN"
  sha256sum -c EVIDENCE.sha256
) >"$audit_dir/resume-evidence-pre.txt"
(
  cd "$RESUME_CHECKPOINT"
  sha256sum -c "$RESUME_RUN/checkpoint-files.sha256"
) >"$audit_dir/resume-checkpoint-pre.txt"

install -m 0444 "$WORKTREE/platform/b300/lifecycle/verify_resume_artifacts.py" "$audit_dir/verify_resume_artifacts.py"
install -m 0444 "$WORKTREE/platform/b300/lifecycle/verify_artifacts.py" "$audit_dir/verify_artifacts.py"
git -C "$WORKTREE" rev-parse HEAD >"$audit_dir/worktree-revision.txt"
git -C "$WORKTREE" status --short >"$audit_dir/worktree-status.txt"
git -C "$WORKTREE" remote -v >"$audit_dir/worktree-remotes.txt"

verify_cmd=(
  /shared/environments/b300/venvs/skyrl-megatron-0f3a2126/bin/python
  "$audit_dir/verify_resume_artifacts.py"
  --result-dir "$RESUME_RUN/results"
  --checkpoint-dir "$RESUME_CHECKPOINT/checkpoints"
  --export-dir "$RESUME_CHECKPOINT/exports"
  --source-result-dir "$SOURCE_RUN/results"
  --source-export-dir "$SOURCE_CHECKPOINT/exports/global_step_1/policy"
  --resume-checkpoint "$SOURCE_CHECKPOINT/checkpoints/global_step_1"
  --source-evidence-pre "$RESUME_RUN/resume-source-evidence-pre.txt"
  --source-evidence-post "$RESUME_RUN/resume-source-evidence-post.txt"
  --source-checkpoint-pre "$RESUME_RUN/resume-source-checkpoint-pre.txt"
  --source-checkpoint-post "$RESUME_RUN/resume-source-checkpoint-post.txt"
  --resume-boundary-record "$RESUME_RUN/resume-boundary-workaround.json"
  --attached-log "$RESUME_RUN/attached.log"
  --post-gpu-processes "$RESUME_RUN/post-selected-gpu-processes.csv"
  --post-containers "$RESUME_RUN/post-containers.txt"
  --post-processes "$RESUME_RUN/post-processes.txt"
  --run-id "$RESUME_RUN_ID"
  --expected-inference-receivers 2
  --inference-log-dir "$RESUME_RUN/inference"
  --output "$audit_dir/artifact-verification.json"
)
printf '%s\0' "${verify_cmd[@]}" | jq -Rs 'split("\u0000")[:-1]' >"$audit_dir/command.json"
PYTHONPATH="$audit_dir" "${verify_cmd[@]}" \
  2> >(tee "$audit_dir/artifact-verification.stderr" >&2) \
  | tee "$audit_dir/artifact-verification.stdout"

(
  cd "$SOURCE_RUN"
  sha256sum -c EVIDENCE.sha256
) >"$audit_dir/source-evidence-post.txt"
(
  cd "$SOURCE_CHECKPOINT"
  sha256sum -c "$SOURCE_RUN/checkpoint-files.sha256"
) >"$audit_dir/source-checkpoint-post.txt"
(
  cd "$RESUME_RUN"
  sha256sum -c EVIDENCE.sha256
) >"$audit_dir/resume-evidence-post.txt"
(
  cd "$RESUME_CHECKPOINT"
  sha256sum -c "$RESUME_RUN/checkpoint-files.sha256"
) >"$audit_dir/resume-checkpoint-post.txt"

for prefix in source-evidence source-checkpoint resume-evidence resume-checkpoint; do
  cmp -s "$audit_dir/${prefix}-pre.txt" "$audit_dir/${prefix}-post.txt" || die "$prefix changed during audit"
done

jq -n \
  --arg audit_run_id "$run_id" \
  --arg source_run_id "$RESUME_RUN_ID" \
  '{audit_run_id:$audit_run_id,original_launcher_status:"FAIL",reason:"captured verifier treated the known nonfatal ModelOpt vLLM plugin cudaDeviceReset import warning as a fatal undefined-symbol error",runtime_status:"TRAINING_COMPLETE",source_run_id:$source_run_id,status:"CORRECTED_OFFLINE_AUDIT_PASS"}' \
  >"$audit_dir/terminal-diagnosis.json"
(
  cd "$audit_dir"
  find . -type f ! -name EVIDENCE.sha256 -print0 | sort -z | xargs -0 sha256sum
) >"$audit_dir/EVIDENCE.sha256"
chmod -R a-w "$audit_dir"
printf 'PASS %s\n' "$run_id"
