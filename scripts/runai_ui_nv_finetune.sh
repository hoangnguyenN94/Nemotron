#!/usr/bin/env bash
set -euo pipefail

# Example entrypoint for running NV-Embed-v2 finetune from the Run:ai UI
# without using the runai CLI wrappers from this repo.
#
# Defaults are tuned for a 2-node distributed job with 8 GPUs per node.
# Run:ai PyTorch workloads usually inject MASTER_ADDR / MASTER_PORT /
# GROUP_RANK / WORLD_SIZE automatically, and this script prefers those
# values when they are available.

WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
HF_CACHE_ROOT="${HF_CACHE_ROOT:-${WORKSPACE_ROOT}/hf-cache}"
TRAIN_DATA_PATH="${TRAIN_DATA_PATH:-${WORKSPACE_ROOT}/output/embed/nv_stage1/train_mined.automodel_unrolled.json}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-${WORKSPACE_ROOT}/output/embed/nv_stage2_multinode/checkpoints}"
NV_BASE_MODEL="${NV_BASE_MODEL:-nvidia/NV-Embed-v2}"
WORKSPACE_STAGE2_DIR="${WORKSPACE_STAGE2_DIR:-${WORKSPACE_ROOT}/nemotron_embed/stage2_finetune}"
IMAGE_STAGE2_DIR="${IMAGE_STAGE2_DIR:-/opt/nemotron/src/nemotron/recipes/embed/stage2_finetune}"
TRAIN_SCRIPT_PATH="${TRAIN_SCRIPT_PATH:-}"

NPROC_PER_NODE="${NPROC_PER_NODE:-${LOCAL_WORLD_SIZE:-8}}"
NNODES="${NNODES:-2}"
NODE_RANK="${NODE_RANK:-${GROUP_RANK:-0}}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

NUM_EPOCHS="${NUM_EPOCHS:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-8}"
LOCAL_BATCH_SIZE="${LOCAL_BATCH_SIZE:-1}"
LEARNING_RATE="${LEARNING_RATE:-1.0e-5}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-1}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
TRAIN_N_PASSAGES="${TRAIN_N_PASSAGES:-5}"
POOLING="${POOLING:-avg}"
L2_NORMALIZE="${L2_NORMALIZE:-true}"
TEMPERATURE="${TEMPERATURE:-0.02}"
QUERY_MAX_LENGTH="${QUERY_MAX_LENGTH:-256}"
PASSAGE_MAX_LENGTH="${PASSAGE_MAX_LENGTH:-256}"
QUERY_PREFIX="${QUERY_PREFIX:-query:}"
PASSAGE_PREFIX="${PASSAGE_PREFIX:-passage:}"
CHECKPOINT_EVERY_STEPS="${CHECKPOINT_EVERY_STEPS:-20}"
VAL_EVERY_STEPS="${VAL_EVERY_STEPS:-20}"

export HF_HOME="${HF_CACHE_ROOT}"
export HUGGINGFACE_HUB_CACHE="${HF_CACHE_ROOT}/hub"
export HF_MODULES_CACHE="${HF_CACHE_ROOT}/modules"
export TRANSFORMERS_CACHE="${HF_CACHE_ROOT}/hub"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NEMOTRON_USE_FSDP="${NEMOTRON_USE_FSDP:-1}"

[ -d "${HUGGINGFACE_HUB_CACHE}" ] || {
  echo "Missing Hugging Face hub cache: ${HUGGINGFACE_HUB_CACHE}" >&2
  exit 1
}

[ -d "${HF_MODULES_CACHE}" ] || {
  echo "Missing Hugging Face modules cache: ${HF_MODULES_CACHE}" >&2
  exit 1
}

[ -f "${TRAIN_DATA_PATH}" ] || {
  echo "Missing training data: ${TRAIN_DATA_PATH}" >&2
  exit 1
}

if [ -z "${TRAIN_SCRIPT_PATH}" ]; then
  if [ -f "${WORKSPACE_STAGE2_DIR}/train.py" ]; then
    TRAIN_SCRIPT_PATH="${WORKSPACE_STAGE2_DIR}/train.py"
  elif [ -f "${IMAGE_STAGE2_DIR}/train.py" ]; then
    TRAIN_SCRIPT_PATH="${IMAGE_STAGE2_DIR}/train.py"
  else
    echo "Missing train.py in both candidate locations:" >&2
    echo "  - ${WORKSPACE_STAGE2_DIR}/train.py" >&2
    echo "  - ${IMAGE_STAGE2_DIR}/train.py" >&2
    echo "Stage the updated stage2_finetune directory onto the PVC or rebuild the image." >&2
    exit 1
  fi
fi

TRAIN_SCRIPT_DIR="$(cd "$(dirname "${TRAIN_SCRIPT_PATH}")" && pwd)"

[ -f "${TRAIN_SCRIPT_DIR}/config/default.yaml" ] || {
  echo "Missing config/default.yaml next to train.py: ${TRAIN_SCRIPT_DIR}/config/default.yaml" >&2
  exit 1
}

[ -f "${TRAIN_SCRIPT_DIR}/biencoder_base.yaml" ] || {
  echo "Missing biencoder_base.yaml next to train.py: ${TRAIN_SCRIPT_DIR}/biencoder_base.yaml" >&2
  exit 1
}

if [ -n "${WORLD_SIZE:-}" ] && [ "${NNODES}" = "2" ] && [ "${NPROC_PER_NODE}" != "0" ]; then
  inferred_nnodes="$((WORLD_SIZE / NPROC_PER_NODE))"
  if [ "${inferred_nnodes}" -ge 1 ]; then
    NNODES="${inferred_nnodes}"
  fi
fi

mkdir -p "${CHECKPOINT_DIR}"

echo "[run.sh] workspace: ${WORKSPACE_ROOT}"
echo "[run.sh] hf cache: ${HF_CACHE_ROOT}"
echo "[run.sh] train data: ${TRAIN_DATA_PATH}"
echo "[run.sh] checkpoints: ${CHECKPOINT_DIR}"
echo "[run.sh] base model: ${NV_BASE_MODEL}"
echo "[run.sh] train script: ${TRAIN_SCRIPT_PATH}"
echo "[run.sh] nnodes=${NNODES} node_rank=${NODE_RANK} nproc_per_node=${NPROC_PER_NODE}"
echo "[run.sh] master=${MASTER_ADDR}:${MASTER_PORT}"
echo "[run.sh] fsdp=${NEMOTRON_USE_FSDP}"

exec run-with-env nv torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  "${TRAIN_SCRIPT_PATH}" \
  train_data_path="${TRAIN_DATA_PATH}" \
  checkpoint_dir="${CHECKPOINT_DIR}" \
  base_model="${NV_BASE_MODEL}" \
  num_epochs="${NUM_EPOCHS}" \
  global_batch_size="${GLOBAL_BATCH_SIZE}" \
  local_batch_size="${LOCAL_BATCH_SIZE}" \
  learning_rate="${LEARNING_RATE}" \
  lr_warmup_steps="${LR_WARMUP_STEPS}" \
  lr_decay_style=cosine \
  weight_decay="${WEIGHT_DECAY}" \
  train_n_passages="${TRAIN_N_PASSAGES}" \
  pooling="${POOLING}" \
  l2_normalize="${L2_NORMALIZE}" \
  temperature="${TEMPERATURE}" \
  query_max_length="${QUERY_MAX_LENGTH}" \
  passage_max_length="${PASSAGE_MAX_LENGTH}" \
  query_prefix="${QUERY_PREFIX}" \
  passage_prefix="${PASSAGE_PREFIX}" \
  checkpoint_every_steps="${CHECKPOINT_EVERY_STEPS}" \
  val_every_steps="${VAL_EVERY_STEPS}"
