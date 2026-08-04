#!/usr/bin/env bash

set -euo pipefail

# Validated fully-async Qwen3-8B production example. Keep its checkpoints,
# W&B state, and exported trajectories isolated from other configurations.

export TINKER_BASE_MODEL="Qwen/Qwen3-8B"
export TINKER_MAX_SEQ_LEN=32768
export TINKER_LORA_RANK=16
export LEARNING_RATE=5.0e-5
export ENABLE_THINKING=false
export TRAIN_BATCH_SIZE=64
export N_SAMPLES_PER_PROMPT=5
export MAX_GENERATE_LENGTH=1024
export MAX_TRAINING_STEPS=50
export MAX_STALENESS_STEPS=4
export NUM_PARALLEL_GENERATION_WORKERS=320
export CKPT_PATH="${CKPT_PATH:-${HOME}/ckpts/gsm8k-tinker-fully-async-qwen3-8b-b64-nonthinking-maxgen1024-lr5e5-production}"
export NUM_LOGGER_TRAIN_SAMPLES=16
export NUM_LOGGER_EVAL_SAMPLES=16
export WANDB_JOB_TYPE="tinker-gsm8k-fully-async-qwen3-8b-lr5e5-nonthinking"
export TRAINER_TAGS="['tinker','gsm8k','qwen3-8b','fully-async','staleness4','nonthinking','maxgen1024','lr5e-5','production']"
export TINKER_PREFILL_PRICE_PER_MILLION=0.195
export TINKER_SAMPLE_PRICE_PER_MILLION=0.60
export TINKER_TRAIN_PRICE_PER_MILLION=0.44

: "${EXPORT_PATH:=${HOME}/exports/gsm8k-tinker-fully-async-qwen3-8b-b64-nonthinking-maxgen1024-lr5e5-production}"
export EXPORT_PATH

run_name_file="${CKPT_PATH}/wandb_run_name.txt"
run_id_file="${CKPT_PATH}/wandb_run_id.txt"
if [[ -z "${WANDB_RUN_ID:-}" && -s "$run_id_file" ]]; then
  IFS= read -r WANDB_RUN_ID < "$run_id_file"
fi
if [[ -z "${RUN_NAME:-}" && -s "$run_name_file" ]]; then
  IFS= read -r RUN_NAME < "$run_name_file"
fi
if [[ -z "${RUN_NAME:-}" || -z "${WANDB_RUN_ID:-}" ]]; then
  run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -z "${RUN_NAME:-}" ]]; then
  RUN_NAME="gsm8k-tinker-fully-async-qwen3-8b-b64n5-s4-nonthinking-maxgen1024-lr5e5-${run_stamp}"
fi
if [[ -z "${WANDB_RUN_ID:-}" ]]; then
  WANDB_RUN_ID="tinker-async-qwen3-8b-lr5e5-nonthinking-${run_stamp}"
fi
export RUN_NAME WANDB_RUN_ID

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${script_dir}/fully_async_run_gsm8k_tinker.sh" \
  trainer.export_path="$EXPORT_PATH" \
  "$@"
