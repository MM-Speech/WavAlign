#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

VARIANT="${1:-plus-vanilla}"
TIMESTAMP="${2:-$(date +'%Y%m%d_%H%M%S')}"
DATASET_PATH="${3:-${REPO_ROOT}/examples/sample_dpo.jsonl}"

VITA_AUDIO_ROOT="${VITA_AUDIO_ROOT:?Please set VITA_AUDIO_ROOT to your local VITA-Audio checkout.}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/outputs}"
OUTPUT_DIR="${OUTPUT_DIR:-${OUTPUT_ROOT}/vita-audio-${VARIANT}-dpo/${TIMESTAMP}}"

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${OUTPUT_DIR}/train.log"
exec &> >(tee -a "${LOG_FILE}")

PRECISION="${PRECISION:-bf16}"
case "${PRECISION}" in
  bf16)
    BF16_FLAG=True
    FP16_FLAG=False
    COMM_DTYPE=bf16
    ;;
  fp16)
    BF16_FLAG=False
    FP16_FLAG=True
    COMM_DTYPE=fp16
    ;;
  *)
    echo "Unsupported PRECISION: ${PRECISION}" >&2
    exit 1
    ;;
esac

case "${VARIANT}" in
  balance)
    MODEL_PATH="${MODEL_NAME_OR_PATH:-${VITA_AUDIO_ROOT}/models/VITA-Audio-Balance}"
    TOKENIZER_TYPE="${AUDIO_TOKENIZER_TYPE:-glm4voice}"
    TEXT_AUDIO_RATIO="${TEXT_AUDIO_INTERVAL_RATIO:-1 4 3 8 4 10}"
    ;;
  boost)
    MODEL_PATH="${MODEL_NAME_OR_PATH:-${VITA_AUDIO_ROOT}/models/VITA-Audio-Boost}"
    TOKENIZER_TYPE="${AUDIO_TOKENIZER_TYPE:-glm4voice}"
    TEXT_AUDIO_RATIO="${TEXT_AUDIO_INTERVAL_RATIO:-1 10 4 10}"
    ;;
  plus-vanilla)
    MODEL_PATH="${MODEL_NAME_OR_PATH:-${VITA_AUDIO_ROOT}/models/VITA-Audio-Plus-Vanilla}"
    TOKENIZER_TYPE="${AUDIO_TOKENIZER_TYPE:-sensevoice_glm4voice}"
    TEXT_AUDIO_RATIO="${TEXT_AUDIO_INTERVAL_RATIO:-1 10 4 10}"
    ;;
  *)
    echo "Unsupported variant: ${VARIANT}" >&2
    exit 1
    ;;
esac

AUDIO_TOKENIZER_PATH="${AUDIO_TOKENIZER_PATH:-${VITA_AUDIO_ROOT}/models/glm-4-voice-tokenizer}"
FLOW_PATH="${FLOW_PATH:-${VITA_AUDIO_ROOT}/models/glm-4-voice-decoder}"
BASE_DS_CONFIG="${BASE_DS_CONFIG:-${REPO_ROOT}/config/ds_config_rl_sft.json}"
TEMP_DS_CONFIG="${OUTPUT_DIR}/ds_config.json"

python - <<'PY' "${BASE_DS_CONFIG}" "${TEMP_DS_CONFIG}" "${BF16_FLAG}" "${FP16_FLAG}" "${COMM_DTYPE}"
import json
import sys

src, dst, bf16_flag, fp16_flag, comm_dtype = sys.argv[1:6]
bf16 = bf16_flag.lower() == "true"
fp16 = fp16_flag.lower() == "true"

with open(src, "r", encoding="utf-8") as handle:
    config = json.load(handle)

config.setdefault("bf16", {})
config.setdefault("fp16", {})
config["bf16"]["enabled"] = bf16
config["fp16"]["enabled"] = fp16
config["communication_data_type"] = comm_dtype

with open(dst, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2)
    handle.write("\n")
PY

export PYTHONPATH="${REPO_ROOT}:${VITA_AUDIO_ROOT}:${VITA_AUDIO_ROOT}/third_party/GLM-4-Voice:${PYTHONPATH:-}"
export VITA_AUDIO_ROOT

torchrun \
  --nproc_per_node "${NPROC_PER_NODE:-1}" \
  --nnodes "${NNODES:-1}" \
  --node_rank "${NODE_RANK:-0}" \
  --master_addr "${MASTER_ADDR:-localhost}" \
  --master_port "${MASTER_PORT:-29502}" \
  "${REPO_ROOT}/train_vita_audio_dpo.py" \
  --model_name_or_path "${MODEL_PATH}" \
  --model_variant "${VARIANT}" \
  --audio_tokenizer_path "${AUDIO_TOKENIZER_PATH}" \
  --audio_tokenizer_type "${TOKENIZER_TYPE}" \
  --flow_path "${FLOW_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --output_dir "${OUTPUT_DIR}" \
  --text_audio_interval_ratio "${TEXT_AUDIO_RATIO}" \
  --max_steps "${MAX_STEPS:-2000}" \
  --per_device_train_batch_size "${BATCH_SIZE:-1}" \
  --gradient_accumulation_steps "${GRAD_ACCUM:-1}" \
  --learning_rate "${LEARNING_RATE:-5e-6}" \
  --warmup_steps "${WARMUP_STEPS:-50}" \
  --lr_scheduler_type "${LR_SCHEDULER_TYPE:-cosine}" \
  --logging_steps "${LOGGING_STEPS:-1}" \
  --save_steps "${SAVE_STEPS:-100}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --bf16 "${BF16_FLAG}" \
  --fp16 "${FP16_FLAG}" \
  --tf32 True \
  --gradient_checkpointing "${GRADIENT_CHECKPOINTING:-True}" \
  --audio_model_freeze "${AUDIO_MODEL_FREEZE:-True}" \
  --task_types "${TASK_TYPES:-s2s}" \
  --dpo_beta "${DPO_BETA:-0.2}" \
  --num_negative_samples "${NUM_NEGATIVE_SAMPLES:-1}" \
  --negative_max_new_tokens "${NEGATIVE_MAX_NEW_TOKENS:-512}" \
  --negative_temperature "${NEGATIVE_TEMPERATURE:-0.7}" \
  --negative_top_p "${NEGATIVE_TOP_P:-0.9}" \
  --deepspeed_config "${TEMP_DS_CONFIG}" \
  "${@:4}"
