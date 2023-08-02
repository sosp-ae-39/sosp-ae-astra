#/bin/bash

set -e

REQUEST_RATE=20.0
DURATION=900

for sampling in "n2" "n4" "n6" "n2-beam" "n4-beam" "n6-beam"; do
    python experiments/text_completion.py \
        --system astra --model opt-13b \
        --duration $DURATION --sampling $sampling \
        --dataset alpaca --rates $REQUEST_RATE \
        --do-memory-analysis
done
