import os
import pickle

import matplotlib.pyplot as plt
import numpy as np


SAVE_DIRS = [
    './results/sharegpt/opt-13b-tp1/n1/orca-constant/block16/req-rate-1.5/seed0/duration-900/',
    './results/sharegpt/opt-13b-tp1/n1/orca-power2/block16/req-rate-1.5/seed0/duration-900/',
    './results/sharegpt/opt-13b-tp1/n1/orca-oracle/block16/req-rate-1.5/seed0/duration-900/',
    './results/sharegpt/opt-13b-tp1/n1/cacheflow/block16/req-rate-1.5/seed0/duration-900/',
]

COLORS = [
    'red',
    'orange',
    'green',
    'blue',
]

SYSTEMS=[
    'Orca\n(Max)',
    'Orca\n(Pow2)',
    'Orca\n(Oracle)',
    'Astra',
]


def plot_memory_usage(xleft: float = 600, xright: float = 3600):
    # Draw a bar chart for each file.
    fig = plt.figure(figsize=(4, 3))

    # plt.bar(0, 0, color=color, label='KVFlow')
    # plt.text(0, 0, '0.00', ha='center', va='bottom', fontsize=12)

    for i in range(len(SAVE_DIRS)):
        with open(os.path.join(SAVE_DIRS[i], 'stats.pkl'), 'rb') as f:
            stats = pickle.load(f)
        timestamps = stats['timestamps']
        for j, t in enumerate(timestamps):
            if t >= xleft:
                start = j
                break
        for j, t in enumerate(timestamps):
            if t > xright:
                end = j
                break
        else:
            end = j

        num_gpu_blocks = stats['num_gpu_blocks']
        block_size = 2 * 2 * 5120 * 40 * 16
        kv_cache_size = num_gpu_blocks * block_size

        gpu_cache_usage = stats['gpu_cache_usage']
        gpu_cache_usage = gpu_cache_usage[start:end]
        avg_gpu_cache_usage = np.mean(gpu_cache_usage)
        avg_gpu_cache_usage = avg_gpu_cache_usage * kv_cache_size
        avg_gpu_cache_usage /= 1024 * 1024 * 1024

        plt.bar(i, avg_gpu_cache_usage, color=COLORS[i], label=SYSTEMS[i])
        plt.text(i, avg_gpu_cache_usage, f'{avg_gpu_cache_usage:.2f}', ha='center', va='bottom', fontsize=12)

    plt.xticks(range(len(SAVE_DIRS)), SYSTEMS, fontsize=12)
    # plt.legend(loc='upper center', fontsize=12, frameon=False, bbox_to_anchor=(0.5, 1.15), ncol=3)
    plt.ylabel('Memory usage (GB)', fontsize=12)
    plt.ylim(0, 13)
    plt.tight_layout()
    os.makedirs('./figures', exist_ok=True)
    plt.savefig('./figures/fig13_a.pdf')


def plot_batch_size(xleft: float = 600, xright: float = 3600):
    # Draw a bar chart for each file.
    fig = plt.figure(figsize=(4, 3))

    # plt.bar(0, 0, color=color, label='KVFlow')
    # plt.text(0, 0, '0.00', ha='center', va='bottom', fontsize=12)

    for i in range(len(SAVE_DIRS)):
        with open(os.path.join(SAVE_DIRS[i], 'stats.pkl'), 'rb') as f:
            stats = pickle.load(f)
        timestamps = stats['timestamps']
        for j, t in enumerate(timestamps):
            if t >= xleft:
                start = j
                break
        for j, t in enumerate(timestamps):
            if t > xright:
                end = j
                break
        else:
            end = j

        num_running = stats['num_running']
        num_running = num_running[start:end]
        avg_num_running = np.mean(num_running)
        plt.bar(i, avg_num_running, color=COLORS[i], label=SYSTEMS[i])
        plt.text(i, avg_num_running, f'{avg_num_running:.2f}', ha='center', va='bottom', fontsize=12)

    plt.xticks(range(len(SAVE_DIRS)), SYSTEMS, fontsize=12)
    plt.ylim(0, 25)
    plt.ylabel('# Batched requests', fontsize=12)
    plt.tight_layout()

    os.makedirs('./figures', exist_ok=True)
    plt.savefig('./figures/fig13_b.pdf')


if __name__ == '__main__':
    plot_memory_usage()
    plot_batch_size()
