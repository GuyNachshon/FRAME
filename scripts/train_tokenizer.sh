#!/usr/bin/env bash
set -euo pipefail
DOMAIN=${1:-vizdoom}
NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
echo "Training tokenizer for domain: ${DOMAIN} on ${NUM_GPUS} GPU(s)"
shift || true
uv run accelerate launch --num_processes="${NUM_GPUS}" --mixed_precision=fp16 -m tokenizer.train --config "configs/${DOMAIN}_tokenizer.yaml" "$@"
