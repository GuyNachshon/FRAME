#!/usr/bin/env bash
set -euo pipefail
DOMAIN=${1:-vizdoom}
echo "Training tokenizer for domain: ${DOMAIN}"
uv run python -m tokenizer.train --config "configs/${DOMAIN}_tokenizer.yaml"
