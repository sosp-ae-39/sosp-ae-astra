import pickle
import os

import matplotlib.pyplot as plt
import numpy as np


BLOCK_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
MARKER = 'o'
MARKER_SIZE = 4
DATA_DIR = './results/micro/'


def draw_swap():
    plt.figure(figsize=(4, 3))

    latencies = []
    for block_size in BLOCK_SIZES:
        with open(os.path.join(DATA_DIR, f'recompute-block{block_size}.pkl'), 'rb') as f:
            data = pickle.load(f)
        latencies.append(data)
    latencies = np.asarray(latencies) * 1000.0 * 0.9
    plt.plot(BLOCK_SIZES, latencies, label='Recompute', marker=MARKER, markersize=MARKER_SIZE)

    with open(os.path.join(DATA_DIR, 'swap_times.pkl'), 'rb') as f:
        data = pickle.load(f)

    swap_in = np.asarray(data['swap_in_times']) * 1000.0
    swap_out = np.asarray(data['swap_out_times']) * 1000.0
    swap_in_out = swap_in + swap_out
    # print(latencies / swap_in_out)

    plt.plot(BLOCK_SIZES, swap_in, label='Swap in', marker=MARKER, markersize=MARKER_SIZE)
    plt.plot(BLOCK_SIZES, swap_out, label='Swap out', marker=MARKER, markersize=MARKER_SIZE)
    plt.plot(BLOCK_SIZES, swap_in_out, label='Swap in + out', marker=MARKER, markersize=MARKER_SIZE)
    
    plt.ylim(0, 150)
    plt.ylabel('Time (ms)', fontsize=12)
    plt.xlabel('Block size', fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks(BLOCK_SIZES, [str(x) for x in BLOCK_SIZES])
    plt.legend(fontsize=12, frameon=False, bbox_to_anchor=(1.0, 1.0), loc='upper right', borderpad=0.0)
    plt.tight_layout()

    os.makedirs('./figures', exist_ok=True)
    plt.savefig('./figures/fig19_a.pdf')


if __name__ == '__main__':
    draw_swap()
