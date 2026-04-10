#!/usr/bin/env bash
set -euo pipefail
DOMAIN=${1:-vizdoom}
TOKENIZER_CKPT=${2:?"Usage: $0 <domain> <tokenizer_checkpoint>"}
echo "Training predictor for domain: ${DOMAIN}"
uv run python -m predictor.train \
    --config "configs/${DOMAIN}_predictor.yaml" \
    --tokenizer_checkpoint "${TOKENIZER_CKPT}"
