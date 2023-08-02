#/bin/bash

set -e

conda activate astra
cd astra

# Execute the micro-benchmark for swapping.
python benchmark/benchmark_swap.py

# Execute the micro-benchmark for recompilation.
for block_size in 1 2 4 8 16 32 64 128 256; do
    python benchmark/benchmark_recomp.py --model facebook/opt-13b --block-size $block_size
done
