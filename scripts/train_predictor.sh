#!/usr/bin/env bash
set -euo pipefail
DOMAIN=${1:-vizdoom}
TOKENIZER_CKPT=${2:?"Usage: $0 <domain> <tokenizer_checkpoint>"}
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "Training predictor for domain: ${DOMAIN} on ${NUM_GPUS} GPU(s)"
shift 2 || true
uv run accelerate launch --num_processes="${NUM_GPUS}" --mixed_precision=fp16 -m predictor.train \
    --config "configs/${DOMAIN}_predictor.yaml" \
    --tokenizer_checkpoint "${TOKENIZER_CKPT}" "$@"
