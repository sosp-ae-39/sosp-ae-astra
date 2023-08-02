#/bin/bash

set -e

REQUEST_RATE=2.01
DURATION=600

# Recomputation: Measure performance of each block size.
for block_size in 1 2 4 8 16 32 64 128 256; do
    python experiments/text_completion.py \
        --system astra --model opt-13b \
        --duration $DURATION --sampling n1 \
        --dataset sharegpt --rates $REQUEST_RATE \
        --block-size $block_size
done

# Swapping: Measure performance of each block size.
for block_size in 1 2 4 8 16 32 64 128 256; do
    python experiments/text_completion.py \
        --system astra --model opt-13b \
        --duration $DURATION --sampling n1 \
        --dataset sharegpt --rates $REQUEST_RATE \
        --block-size $block_size --always-swap
done
