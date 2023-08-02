import torch
from cacheflow import cache_ops
import numpy as np

import os
import pickle
import time
import random


def benchmark_swap(
    mode: str,
    num_mappings: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    block_size: int,
    num_blocks: int,
    dtype: torch.dtype,
) -> None:
    # Generate random block mappings.
    src_blocks = random.sample(range(num_blocks), num_mappings)
    dst_blocks = random.sample(range(num_blocks), num_mappings)
    block_mapping = {src: dst for src, dst in zip(src_blocks, dst_blocks)}

    # Create the KV cache.
    x = 16 // torch.tensor([], dtype=dtype).element_size()
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    gpu_key_caches = []
    cpu_key_caches = []
    for _ in range(num_layers):
        gpu_key_cache = torch.randn(
            size=key_cache_shape, dtype=dtype, device='cuda')
        gpu_key_caches.append(gpu_key_cache)
        cpu_key_cache = torch.empty(
            size=key_cache_shape, dtype=dtype, device='cpu', pin_memory=True)
        cpu_key_caches.append(cpu_key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    gpu_value_caches = []
    cpu_value_caches = []
    for _ in range(num_layers):
        gpu_value_cache = torch.randn(
            size=value_cache_shape, dtype=dtype, device='cuda')
        gpu_value_caches.append(gpu_value_cache)
        cpu_value_cache = torch.empty(
            size=value_cache_shape, dtype=dtype, device='cpu', pin_memory=True)
        cpu_value_caches.append(cpu_value_cache)

    def swap():
        for i in range(num_layers):
            if mode == 'in':
                src_key_cache = cpu_key_caches[i]
                dst_key_cache = gpu_key_caches[i]
                src_value_cache = cpu_value_caches[i]
                dst_value_cache = gpu_value_caches[i]
            elif mode == 'out':
                src_key_cache = gpu_key_caches[i]
                dst_key_cache = cpu_key_caches[i]
                src_value_cache = gpu_value_caches[i]
                dst_value_cache = cpu_value_caches[i]
            else:
                assert False
            # Copy the key blocks.
            cache_ops.swap_blocks(
                src_key_cache, dst_key_cache, block_mapping)
            # Copy the value blocks.
            cache_ops.swap_blocks(
                src_value_cache, dst_value_cache, block_mapping)

    for _ in range(3):
        swap()
    torch.cuda.synchronize()

    swap_times = []
    for _ in range(10):
        start = time.time()
        swap()
        torch.cuda.synchronize()
        end = time.time()
        swap_times.append(end - start)
    return np.mean(swap_times)


if __name__ == '__main__':
    OUTPUT_DIR = '../results/micro/'
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    NUM_LAYERS = 40
    NUM_HEADS = 40
    HEAD_SIZE = 128

    swap_in_times = []
    swap_out_times = []

    TOTAL_SLOTS = 12 * 1024
    for block_size in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        num_blocks = TOTAL_SLOTS // block_size
        num_mappings = 256 // block_size
        for mode in ['out', 'in']:
            t = benchmark_swap(
                mode=mode,
                num_mappings=num_mappings,
                num_layers=NUM_LAYERS,
                num_heads=NUM_HEADS,
                head_size=HEAD_SIZE,
                block_size=block_size,
                num_blocks=num_blocks,
                dtype=torch.float16,
            )
            if mode == 'in':
                swap_in_times.append(t)
            elif mode == 'out':
                swap_out_times.append(t)

            with open(os.path.join(OUTPUT_DIR, 'swap_times.pkl'), 'wb') as f:
                pickle.dump({
                    'swap_in_times': swap_in_times,
                    'swap_out_times': swap_out_times,
                }, f)
