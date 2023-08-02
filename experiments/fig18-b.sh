#/bin/bash

set -e

REQUEST_RATE=30.1
DURATION=600

# Measure performance of each block size.
for block_size in 1 2 4 8 16 32 64 128 256; do
    python experiments/text_completion.py \
        --system astra --model opt-13b \
        --duration $DURATION --sampling n1 \
        --dataset alpaca --rates $REQUEST_RATE \
        --block-size $block_size
done
