#!/usr/bin/env bash

set -euo pipefail

# Production-oriented synchronous hosted-Tinker GRPO. The paid-run guard and
# wall-clock timeout stay enabled for both short validation and full runs.
#
# Prepare data first:
#   python examples/train/gsm8k/gsm8k_dataset.py --output_dir "$HOME/data/gsm8k"

: "${DATA_DIR:=${HOME}/data/gsm8k}"
: "${TINKER_BASE_MODEL:=Qwen/Qwen3.5-4B}"
: "${TINKER_MAX_SEQ_LEN:=32768}"
: "${TINKER_LORA_RANK:=16}"
: "${TRAIN_BATCH_SIZE:=64}"
: "${N_SAMPLES_PER_PROMPT:=5}"
: "${MAX_GENERATE_LENGTH:=1024}"
: "${ENABLE_THINKING:=true}"
: "${TRAIN_TEMPERATURE:=1.0}"
: "${TRAIN_TOP_P:=1.0}"
: "${TRAIN_TOP_K:=-1}"
: "${EVAL_TEMPERATURE:=0.0}"
: "${EVAL_TOP_P:=1.0}"
: "${EVAL_TOP_K:=-1}"
: "${ZERO_REWARD_ON_NON_STOP:=false}"
: "${MAX_TRAINING_STEPS:=50}"
: "${TRAINING_EPOCHS:=20}"
: "${LEARNING_RATE:=1.0e-6}"
: "${EVAL_BATCH_SIZE:=1024}"
: "${EVAL_INTERVAL:=10}"
: "${NUM_LOGGER_TRAIN_SAMPLES:=-1}"
: "${NUM_LOGGER_EVAL_SAMPLES:=-1}"
: "${CKPT_INTERVAL:=10}"
: "${CKPT_PATH:=${HOME}/ckpts/gsm8k-tinker-sync-b64-thinking-production}"
: "${MAX_CKPTS_TO_KEEP:=7}"
: "${TINKER_CHECKPOINT_TTL_SECONDS:=2592000}"
: "${TINKER_PREFILL_PRICE_PER_MILLION:=0.33}"
: "${TINKER_SAMPLE_PRICE_PER_MILLION:=1.005}"
: "${TINKER_TRAIN_PRICE_PER_MILLION:=0.737}"
: "${RESUME_MODE:=latest}"
: "${RESUME_PATH:=null}"
: "${MAX_PAID_RUNTIME_MINUTES:=180}"
: "${LOGGER:=wandb}"
: "${PROJECT_NAME:=skyrl}"
: "${WANDB_ENTITY:=}"
: "${WANDB_API_KEY_FILE:=}"
: "${WANDB_MODE:=online}"
: "${WANDB_JOB_TYPE:=tinker-gsm8k-sync}"

if [[ "$ENABLE_THINKING" == "true" ]]; then
  thinking_tag="thinking"
elif [[ "$ENABLE_THINKING" == "false" ]]; then
  thinking_tag="nonthinking"
else
  printf 'ENABLE_THINKING must be true or false, got: %s\n' "$ENABLE_THINKING" >&2
  exit 2
fi
: "${TRAINER_TAGS:=['tinker','gsm8k','sync','${thinking_tag}','maxgen${MAX_GENERATE_LENGTH}','production']}"
if [[ "$ZERO_REWARD_ON_NON_STOP" != "true" && "$ZERO_REWARD_ON_NON_STOP" != "false" ]]; then
  printf 'ZERO_REWARD_ON_NON_STOP must be true or false, got: %s\n' \
    "$ZERO_REWARD_ON_NON_STOP" >&2
  exit 2
fi

run_stamp="$(date -u +%Y%m%dT%H%M%SZ)"
wandb_run_id_file="${CKPT_PATH}/wandb_run_id.txt"
wandb_run_name_file="${CKPT_PATH}/wandb_run_name.txt"
if [[ "$LOGGER" == "wandb" && "$RESUME_MODE" == "latest" ]]; then
  if [[ -z "${WANDB_RUN_ID:-}" && -s "$wandb_run_id_file" ]]; then
    IFS= read -r WANDB_RUN_ID < "$wandb_run_id_file"
  fi
  if [[ -z "${RUN_NAME:-}" && -s "$wandb_run_name_file" ]]; then
    IFS= read -r RUN_NAME < "$wandb_run_name_file"
  fi
fi
: "${RUN_NAME:=gsm8k-tinker-sync-qwen35-4b-b${TRAIN_BATCH_SIZE}n${N_SAMPLES_PER_PROMPT}-${thinking_tag}-maxgen${MAX_GENERATE_LENGTH}-${run_stamp}}"
: "${WANDB_RUN_ID:=tinker-sync-${run_stamp}}"
: "${WANDB_RESUME:=allow}"
: "${WANDB_DIR:="${CKPT_PATH}/wandb"}"
wandb_target="${PROJECT_NAME}"
if [[ -n "$WANDB_ENTITY" ]]; then
  wandb_target="${WANDB_ENTITY}/${PROJECT_NAME}"
fi

for positive_integer in \
  "$TRAIN_BATCH_SIZE" \
  "$N_SAMPLES_PER_PROMPT" \
  "$MAX_GENERATE_LENGTH" \
  "$MAX_TRAINING_STEPS" \
  "$TRAINING_EPOCHS" \
  "$EVAL_BATCH_SIZE" \
  "$EVAL_INTERVAL" \
  "$CKPT_INTERVAL" \
  "$MAX_CKPTS_TO_KEEP" \
  "$TINKER_CHECKPOINT_TTL_SECONDS" \
  "$MAX_PAID_RUNTIME_MINUTES"; do
  if [[ ! "$positive_integer" =~ ^[1-9][0-9]*$ ]]; then
    printf 'Expected a positive integer, got: %s\n' "$positive_integer" >&2
    exit 2
  fi
done

for logger_sample_count in \
  "$NUM_LOGGER_TRAIN_SAMPLES" \
  "$NUM_LOGGER_EVAL_SAMPLES"; do
  if [[ ! "$logger_sample_count" =~ ^-1$|^[0-9]+$ ]]; then
    printf 'Logger sample counts must be -1 or a non-negative integer, got: %s\n' \
      "$logger_sample_count" >&2
    exit 2
  fi
done

for top_k in "$TRAIN_TOP_K" "$EVAL_TOP_K"; do
  if [[ ! "$top_k" =~ ^-1$|^[1-9][0-9]*$ ]]; then
    printf 'Top-k must be -1 or a positive integer, got: %s\n' "$top_k" >&2
    exit 2
  fi
done

for nonnegative_number in \
  "$TINKER_PREFILL_PRICE_PER_MILLION" \
  "$TINKER_SAMPLE_PRICE_PER_MILLION" \
  "$TINKER_TRAIN_PRICE_PER_MILLION"; do
  if [[ ! "$nonnegative_number" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    printf 'Expected a non-negative decimal price, got: %s\n' "$nonnegative_number" >&2
    exit 2
  fi
done

for data_file in "$DATA_DIR/train.parquet" "$DATA_DIR/validation.parquet"; do
  if [[ ! -s "$data_file" ]]; then
    printf 'Required GSM8K data file is missing or empty: %s\n' "$data_file" >&2
    exit 2
  fi
done

if [[ "${TINKER_RUN_CONFIRMED:-0}" != "1" ]]; then
  printf '%s\n' \
    "This command opens a real hosted Tinker training session and may cost money." \
    "Resolved synchronous plan:" \
    "  model: ${TINKER_BASE_MODEL}" \
    "  LoRA rank: ${TINKER_LORA_RANK}" \
    "  optimizer steps: ${MAX_TRAINING_STEPS}" \
    "  prompt groups per step: ${TRAIN_BATCH_SIZE}" \
    "  completions per prompt: ${N_SAMPLES_PER_PROMPT}" \
    "  thinking mode: ${ENABLE_THINKING}" \
    "  train sampling: temperature ${TRAIN_TEMPERATURE}, top-p ${TRAIN_TOP_P}, top-k ${TRAIN_TOP_K}" \
    "  eval sampling: temperature ${EVAL_TEMPERATURE}, top-p ${EVAL_TOP_P}, top-k ${EVAL_TOP_K}" \
    "  zero reward when generation does not stop normally: ${ZERO_REWARD_ON_NON_STOP}" \
    "  maximum generated tokens per step: $(( TRAIN_BATCH_SIZE * N_SAMPLES_PER_PROMPT * MAX_GENERATE_LENGTH ))" \
    "  evaluation: before training and every ${EVAL_INTERVAL} steps" \
    "  W&B trajectory samples: ${NUM_LOGGER_TRAIN_SAMPLES} train and ${NUM_LOGGER_EVAL_SAMPLES} eval per logging call" \
    "  checkpoint: every ${CKPT_INTERVAL} steps and at shutdown (${CKPT_PATH}; keep ${MAX_CKPTS_TO_KEEP} local copies; provider TTL ${TINKER_CHECKPOINT_TTL_SECONDS}s)" \
    "  resume: ${RESUME_MODE} (${RESUME_PATH})" \
    "  W&B: ${wandb_target} (${RUN_NAME}; id ${WANDB_RUN_ID}; mode ${WANDB_MODE})" \
    "  token prices per million: prefill \$${TINKER_PREFILL_PRICE_PER_MILLION}, sample \$${TINKER_SAMPLE_PRICE_PER_MILLION}, train \$${TINKER_TRAIN_PRICE_PER_MILLION}" \
    "  usage logging: cumulative wall time, requests, tokens, service times, checkpoints, and estimated token cost" \
    "  wall-clock cap: ${MAX_PAID_RUNTIME_MINUTES} minutes" \
    "  HF export: disabled" \
    "Review the code and plan, then rerun with TINKER_RUN_CONFIRMED=1."
  exit 2
fi

if [[ -z "${TINKER_API_KEY:-}" ]]; then
  printf 'TINKER_API_KEY is not set in this shell.\n' >&2
  exit 2
fi
if [[ "$LOGGER" == "wandb" ]]; then
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    if [[ -z "$WANDB_API_KEY_FILE" || ! -s "$WANDB_API_KEY_FILE" ]]; then
      printf 'Set WANDB_API_KEY or WANDB_API_KEY_FILE for W&B logging.\n' >&2
      exit 2
    fi
    WANDB_API_KEY="$(< "$WANDB_API_KEY_FILE")"
    export WANDB_API_KEY
  fi
  mkdir -p "$CKPT_PATH" "$WANDB_DIR"
  printf '%s\n' "$WANDB_RUN_ID" > "$wandb_run_id_file"
  printf '%s\n' "$RUN_NAME" > "$wandb_run_name_file"
  if [[ -n "$WANDB_ENTITY" ]]; then
    export WANDB_ENTITY
  else
    unset WANDB_ENTITY
  fi
  export WANDB_DIR WANDB_JOB_TYPE WANDB_MODE WANDB_PROJECT="$PROJECT_NAME"
  export WANDB_NAME="$RUN_NAME" WANDB_RESUME WANDB_RUN_ID
fi

set -x
python examples/train/run_with_timeout.py \
  --timeout-seconds="$(( MAX_PAID_RUNTIME_MINUTES * 60 ))" \
  --shutdown-grace-seconds=120 \
  -- \
  python -m skyrl.train.entrypoints.main_tinker \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.strategy=tinker \
  trainer.tinker.base_model="$TINKER_BASE_MODEL" \
  trainer.tinker.max_seq_len="$TINKER_MAX_SEQ_LEN" \
  trainer.tinker.request_timeout_s=600 \
  trainer.tinker.sampling_timeout_s=300 \
  trainer.tinker.close_timeout_s=300 \
  trainer.tinker.checkpoint_ttl_seconds="$TINKER_CHECKPOINT_TTL_SECONDS" \
  trainer.tinker.prefill_price_per_million_tokens="$TINKER_PREFILL_PRICE_PER_MILLION" \
  trainer.tinker.sample_price_per_million_tokens="$TINKER_SAMPLE_PRICE_PER_MILLION" \
  trainer.tinker.train_price_per_million_tokens="$TINKER_TRAIN_PRICE_PER_MILLION" \
  trainer.policy.model.path="$TINKER_BASE_MODEL" \
  trainer.policy.model.lora.rank="$TINKER_LORA_RANK" \
  trainer.policy.optimizer_config.lr="$LEARNING_RATE" \
  trainer.policy.optimizer_config.adam_betas="[0.9,0.95]" \
  trainer.policy.optimizer_config.weight_decay=0.0 \
  trainer.policy.optimizer_config.max_grad_norm=1.0 \
  trainer.policy.optimizer_config.num_warmup_steps=0 \
  trainer.policy.optimizer_config.scheduler=constant_with_warmup \
  trainer.algorithm.advantage_estimator=grpo \
  trainer.algorithm.policy_loss_type=rollout_is \
  trainer.algorithm.use_kl_loss=false \
  trainer.algorithm.use_kl_in_reward=false \
  trainer.algorithm.zero_variance_filter=true \
  trainer.critic.model.path=null \
  trainer.placement.colocate_all=false \
  trainer.placement.colocate_policy_ref=false \
  trainer.epochs="$TRAINING_EPOCHS" \
  trainer.max_training_steps="$MAX_TRAINING_STEPS" \
  trainer.train_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.policy_mini_batch_size="$TRAIN_BATCH_SIZE" \
  trainer.micro_forward_batch_size_per_gpu="$(( TRAIN_BATCH_SIZE * N_SAMPLES_PER_PROMPT ))" \
  trainer.micro_train_batch_size_per_gpu="$(( TRAIN_BATCH_SIZE * N_SAMPLES_PER_PROMPT ))" \
  trainer.update_epochs_per_batch=1 \
  trainer.max_prompt_length=512 \
  trainer.eval_batch_size="$EVAL_BATCH_SIZE" \
  trainer.eval_before_train=true \
  trainer.eval_interval="$EVAL_INTERVAL" \
  trainer.num_logger_train_samples="$NUM_LOGGER_TRAIN_SAMPLES" \
  trainer.num_logger_eval_samples="$NUM_LOGGER_EVAL_SAMPLES" \
  trainer.ckpt_interval="$CKPT_INTERVAL" \
  trainer.ckpt_path="$CKPT_PATH" \
  trainer.max_ckpts_to_keep="$MAX_CKPTS_TO_KEEP" \
  trainer.hf_save_interval=-1 \
  trainer.resume_mode="$RESUME_MODE" \
  trainer.resume_path="$RESUME_PATH" \
  trainer.enable_ray_gpu_monitor=false \
  generator.inference_engine.backend=tinker \
  generator.inference_engine.run_engines_locally=false \
  generator.inference_engine.enable_ray_prometheus_stats=false \
  generator.batched=false \
  generator.max_turns=1 \
  generator.max_input_length=512 \
  generator.chat_template_kwargs.enable_thinking="$ENABLE_THINKING" \
  generator.zero_reward_on_non_stop="$ZERO_REWARD_ON_NON_STOP" \
  generator.n_samples_per_prompt="$N_SAMPLES_PER_PROMPT" \
  generator.sampling_params.max_generate_length="$MAX_GENERATE_LENGTH" \
  generator.sampling_params.temperature="$TRAIN_TEMPERATURE" \
  generator.sampling_params.top_p="$TRAIN_TOP_P" \
  generator.sampling_params.top_k="$TRAIN_TOP_K" \
  generator.sampling_params.logprobs=1 \
  generator.eval_n_samples_per_prompt=1 \
  generator.eval_sampling_params.max_generate_length="$MAX_GENERATE_LENGTH" \
  generator.eval_sampling_params.temperature="$EVAL_TEMPERATURE" \
  generator.eval_sampling_params.top_p="$EVAL_TOP_P" \
  generator.eval_sampling_params.top_k="$EVAL_TOP_K" \
  generator.eval_sampling_params.logprobs=1 \
  environment.env_class=gsm8k \
  trainer.logger="$LOGGER" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.run_name="$RUN_NAME" \
  trainer.tags="$TRAINER_TAGS" \
  "$@"
