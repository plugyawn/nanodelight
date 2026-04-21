#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/results/kondo_v2_family}"
SEEDS="${SEEDS:-42 43}"
EVAL_PROFILE="${EVAL_PROFILE:-gsm8k}"
APPLY_PATCHES="${APPLY_PATCHES:-0}"
UV_NO_SYNC="${UV_NO_SYNC:-0}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
TOKENIZER_NAME="${TOKENIZER_NAME:-$MODEL_NAME}"
PROMPT_FILE="${PROMPT_FILE:-examples/prompts/cot.txt}"
MAX_STEPS="${MAX_STEPS:-8}"
NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-2}"
NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-4}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-8}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
LOGPROB_BATCH_SIZE="${LOGPROB_BATCH_SIZE:-1}"
MAX_TOTAL_SEQUENCE_LENGTH="${MAX_TOTAL_SEQUENCE_LENGTH:-768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-320}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-768}"
VAL_PERIOD="${VAL_PERIOD:-4}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-256}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
KONDO_TARGET_BACKWARD_TOKEN_FRACTION="${KONDO_TARGET_BACKWARD_TOKEN_FRACTION:-0.7}"

mkdir -p "$RUN_ROOT"

run_arm() {
  local seed="$1"
  local name="$2"
  local enabled="$3"
  local mode="$4"
  local recall_floor="$5"
  local run_dir="$RUN_ROOT/seed${seed}/${name}"
  mkdir -p "$run_dir"

  echo "START seed=${seed} arm=${name} $(date -Is)" | tee -a "$RUN_ROOT/launch.log"
  env \
    APPLY_PATCHES="$APPLY_PATCHES" \
    UV_NO_SYNC="$UV_NO_SYNC" \
    EVAL_PROFILE="$EVAL_PROFILE" \
    SEED="$seed" \
    RUN_DIR="$run_dir" \
    MODEL_NAME="$MODEL_NAME" \
    TOKENIZER_NAME="$TOKENIZER_NAME" \
    PROMPT_FILE="$PROMPT_FILE" \
    MAX_STEPS="$MAX_STEPS" \
    NUM_PROMPTS_PER_STEP="$NUM_PROMPTS_PER_STEP" \
    NUM_GENERATIONS_PER_PROMPT="$NUM_GENERATIONS_PER_PROMPT" \
    TRAIN_GLOBAL_BATCH_SIZE="$TRAIN_GLOBAL_BATCH_SIZE" \
    TRAIN_MICRO_BATCH_SIZE="$TRAIN_MICRO_BATCH_SIZE" \
    LOGPROB_BATCH_SIZE="$LOGPROB_BATCH_SIZE" \
    MAX_TOTAL_SEQUENCE_LENGTH="$MAX_TOTAL_SEQUENCE_LENGTH" \
    MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
    VAL_PERIOD="$VAL_PERIOD" \
    MAX_VAL_SAMPLES="$MAX_VAL_SAMPLES" \
    VAL_BATCH_SIZE="$VAL_BATCH_SIZE" \
    KONDO_ENABLED="$enabled" \
    KONDO_MODE="$mode" \
    KONDO_TARGET_BACKWARD_TOKEN_FRACTION="$KONDO_TARGET_BACKWARD_TOKEN_FRACTION" \
    KONDO_RECALL_FLOOR="$recall_floor" \
    bash "$ROOT_DIR/runs/nemo_grpo_kondo_eval.sh" \
    2>&1 | tee "$run_dir/launcher.log"
  echo "DONE seed=${seed} arm=${name} $(date -Is)" | tee -a "$RUN_ROOT/launch.log"
}

: > "$RUN_ROOT/launch.log"
for seed in $SEEDS; do
  run_arm "$seed" baseline 0 off 0.0
  run_arm "$seed" dense_0p7 1 dense_reference 0.0
  run_arm "$seed" oracle_dense_rows_0p7 1 oracle_dense_rows 0.0
  run_arm "$seed" response_dense_rows_v2_0p7 1 response_dense_rows_v2 1.0
done
