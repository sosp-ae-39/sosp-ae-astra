#/bin/bash

set -e

REQUEST_RATE=4.0
DURATION=900

SYSTEMS=(
    "astra"
    "orca-oracle"
    "orca-power2"
    "orca-constant"
)
for system in "${SYSTEMS[@]}"; do
    python experiments/text_completion.py \
        --system $system --model opt-13b \
        --duration $DURATION --sampling n1 \
        --dataset sharegpt --rates $REQUEST_RATE \
        --do-memory-analysis
done
