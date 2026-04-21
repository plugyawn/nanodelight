#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NEMO_RL_ROOT="${NEMO_RL_ROOT:-$ROOT_DIR/tmp/NeMo-RL}"

if [[ ! -d "$NEMO_RL_ROOT" ]]; then
  echo "NeMo-RL checkout not found at $NEMO_RL_ROOT" >&2
  echo "Set NEMO_RL_ROOT=/path/to/NeMo-RL and retry." >&2
  exit 1
fi

APPLY_PATCHES="${APPLY_PATCHES:-1}"
DRY_RUN="${DRY_RUN:-0}"
EVAL_PROFILE="${EVAL_PROFILE:-gsm8k}"
SEED="${SEED:-42}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
TOKENIZER_NAME="${TOKENIZER_NAME:-$MODEL_NAME}"
PROMPT_FILE="${PROMPT_FILE:-examples/prompts/cot.txt}"
HF_HOME="${HF_HOME:-$NEMO_RL_ROOT/.hf-cache}"

KONDO_ENABLED="${KONDO_ENABLED:-1}"
KONDO_MODE="${KONDO_MODE:-response_dense_rows_v2}"
KONDO_TARGET_BACKWARD_TOKEN_FRACTION="${KONDO_TARGET_BACKWARD_TOKEN_FRACTION:-0.7}"
KONDO_RECALL_FLOOR="${KONDO_RECALL_FLOOR:-1.0}"
KONDO_PRIORITY_MODE="${KONDO_PRIORITY_MODE:-delight}"
KONDO_BLOCK_SIZE="${KONDO_BLOCK_SIZE:-8}"
KONDO_MIN_SELECTED_ROWS="${KONDO_MIN_SELECTED_ROWS:-1}"
KONDO_BYPASS_ON_NONFINITE="${KONDO_BYPASS_ON_NONFINITE:-true}"

MAX_STEPS="${MAX_STEPS:-8}"
NUM_PROMPTS_PER_STEP="${NUM_PROMPTS_PER_STEP:-2}"
NUM_GENERATIONS_PER_PROMPT="${NUM_GENERATIONS_PER_PROMPT:-4}"
TRAIN_GLOBAL_BATCH_SIZE="${TRAIN_GLOBAL_BATCH_SIZE:-8}"
TRAIN_MICRO_BATCH_SIZE="${TRAIN_MICRO_BATCH_SIZE:-1}"
LOGPROB_BATCH_SIZE="${LOGPROB_BATCH_SIZE:-1}"
MAX_TOTAL_SEQUENCE_LENGTH="${MAX_TOTAL_SEQUENCE_LENGTH:-768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-320}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-$MAX_TOTAL_SEQUENCE_LENGTH}"
VAL_PERIOD="${VAL_PERIOD:-4}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-256}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-64}"
MATH_NUM_WORKERS="${MATH_NUM_WORKERS:-4}"
REFERENCE_POLICY_KL_PENALTY="${REFERENCE_POLICY_KL_PENALTY:-}"

if [[ "$KONDO_ENABLED" == "1" ]]; then
  RUN_FLAVOR="kondo_${KONDO_MODE}_rho${KONDO_TARGET_BACKWARD_TOKEN_FRACTION}"
else
  RUN_FLAVOR="baseline"
fi

RUN_NAME="${RUN_NAME:-${EVAL_PROFILE}_${RUN_FLAVOR}_seed${SEED}}"
RUN_DIR="${RUN_DIR:-$NEMO_RL_ROOT/results/$RUN_NAME}"

export HF_HOME
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export NEMO_RL_PY_EXECUTABLES_SYSTEM="${NEMO_RL_PY_EXECUTABLES_SYSTEM:-1}"

UV_BIN="${UV_BIN:-}"
if [[ -z "$UV_BIN" && "${DRY_RUN}" == "1" ]]; then
  UV_BIN="uv"
fi
if [[ -z "$UV_BIN" ]]; then
  UV_BIN="$(command -v uv || true)"
fi
if [[ -z "$UV_BIN" ]]; then
  echo "uv not found on PATH. Set UV_BIN=/path/to/uv and retry." >&2
  exit 1
fi

UV_RUN=("$UV_BIN" run)
if [[ "${UV_NO_SYNC:-0}" == "1" ]]; then
  UV_RUN+=(--no-sync)
fi
UV_RUN+=(python)

patch_already_applied() {
  local patch_path="$1"
  git -C "$NEMO_RL_ROOT" apply --reverse --check "$patch_path" >/dev/null 2>&1
}

apply_patch_once() {
  local patch_path="$1"
  if patch_already_applied "$patch_path"; then
    echo "Patch already applied: $patch_path"
  else
    git -C "$NEMO_RL_ROOT" apply --3way "$patch_path"
  fi
}

if [[ "$APPLY_PATCHES" == "1" ]]; then
  apply_patch_once "$ROOT_DIR/dev/nemo_rl/patches/0001-dg-grpo-loss-and-metrics.patch"
  apply_patch_once "$ROOT_DIR/dev/nemo_rl/patches/0002-dtensor-tokenizer-bootstrap.patch"
  apply_patch_once "$ROOT_DIR/dev/nemo_rl/patches/0003-kondo-routing.patch"
fi

mkdir -p "$RUN_DIR"

DATA_OVERRIDES=("data.train.split_validation_size=0.0")
case "$EVAL_PROFILE" in
  gsm8k)
    DATA_OVERRIDES+=(
      "data.train.dataset_name=gsm8k"
      "+data.train.split=train"
      "++data.validation.dataset_name=gsm8k"
      "++data.validation.split=test"
    )
    ;;
  aime2024)
    DATA_OVERRIDES+=(
      "data.train.dataset_name=OpenMathInstruct-2"
      "++data.validation.dataset_name=AIME2024"
      "++data.validation.repeat=16"
    )
    ;;
  *)
    echo "Unsupported EVAL_PROFILE=$EVAL_PROFILE. Use gsm8k or aime2024." >&2
    exit 1
    ;;
esac

CMD=(
  "${UV_RUN[@]}" examples/run_grpo.py
  "logger.log_dir=${RUN_DIR}/logs"
  "checkpointing.checkpoint_dir=${RUN_DIR}/checkpoints"
  "grpo.max_num_steps=${MAX_STEPS}"
  "grpo.num_prompts_per_step=${NUM_PROMPTS_PER_STEP}"
  "grpo.num_generations_per_prompt=${NUM_GENERATIONS_PER_PROMPT}"
  "grpo.val_period=${VAL_PERIOD}"
  "grpo.val_at_end=true"
  "grpo.max_val_samples=${MAX_VAL_SAMPLES}"
  "grpo.val_batch_size=${VAL_BATCH_SIZE}"
  "grpo.seed=${SEED}"
  "policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE}"
  "policy.train_micro_batch_size=${TRAIN_MICRO_BATCH_SIZE}"
  "policy.logprob_batch_size=${LOGPROB_BATCH_SIZE}"
  "policy.max_total_sequence_length=${MAX_TOTAL_SEQUENCE_LENGTH}"
  "policy.model_name=${MODEL_NAME}"
  "policy.tokenizer.name=${TOKENIZER_NAME}"
  "policy.generation.max_new_tokens=${MAX_NEW_TOKENS}"
  "policy.generation.vllm_cfg.max_model_len=${VLLM_MAX_MODEL_LEN}"
  "policy.generation.vllm_cfg.enable_vllm_metrics_logger=false"
  "checkpointing.enabled=false"
  "checkpointing.keep_top_k=1"
  "checkpointing.save_period=1000000"
  "data.num_workers=1"
  "env.math.num_workers=${MATH_NUM_WORKERS}"
  "logger.tensorboard_enabled=true"
  "logger.wandb_enabled=false"
  "logger.mlflow_enabled=false"
  "logger.swanlab_enabled=false"
  "logger.monitor_gpus=true"
  "logger.num_val_samples_to_print=0"
  "data.default.prompt_file=${PROMPT_FILE}"
  "policy.dtensor_cfg._v2=false"
  "checkpointing.model_save_format=null"
  "loss_fn.dg_enabled=false"
  "grpo.kondo.enabled=${KONDO_ENABLED}"
  "grpo.kondo.mode=${KONDO_MODE}"
  "grpo.kondo.target_backward_token_fraction=${KONDO_TARGET_BACKWARD_TOKEN_FRACTION}"
  "grpo.kondo.recall_floor=${KONDO_RECALL_FLOOR}"
  "grpo.kondo.priority_mode=${KONDO_PRIORITY_MODE}"
  "grpo.kondo.min_selected_rows=${KONDO_MIN_SELECTED_ROWS}"
  "grpo.kondo.bypass_on_nonfinite=${KONDO_BYPASS_ON_NONFINITE}"
)
case "$KONDO_MODE" in
  block_dense_cover|response_block_rows|oracle_block_rows)
    CMD+=("grpo.kondo.block_size=${KONDO_BLOCK_SIZE}")
    ;;
esac
if [[ -n "$REFERENCE_POLICY_KL_PENALTY" ]]; then
  CMD+=("loss_fn.reference_policy_kl_penalty=${REFERENCE_POLICY_KL_PENALTY}")
fi
CMD+=("${DATA_OVERRIDES[@]}")

printf '%s\n' "${CMD[@]}" > "${RUN_DIR}/command.txt"

if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY_RUN=1, command written to %s\n' "${RUN_DIR}/command.txt"
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

(
  cd "$NEMO_RL_ROOT"
  "${CMD[@]}" 2>&1 | tee "${RUN_DIR}/stdout.log"
)
