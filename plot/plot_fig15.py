import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

PARALLEL_GEN_DIRS = [
    './results/alpaca/opt-13b-tp1/n2/cacheflow/block16/req-rate-20.0/seed0/duration-900/',
    './results/alpaca/opt-13b-tp1/n4/cacheflow/block16/req-rate-20.0/seed0/duration-900/',
    './results/alpaca/opt-13b-tp1/n6/cacheflow/block16/req-rate-20.0/seed0/duration-900/',
]

BEAM_DIRS = [
    './results/alpaca/opt-13b-tp1/n2-beam/cacheflow/block16/req-rate-20.0/seed0/duration-900/',
    './results/alpaca/opt-13b-tp1/n4-beam/cacheflow/block16/req-rate-20.0/seed0/duration-900/',
    './results/alpaca/opt-13b-tp1/n6-beam/cacheflow/block16/req-rate-20.0/seed0/duration-900/',
]


def plot(stat_dirs, color, xleft: float = 300, xright: float = 900):
    # Draw a bar chart for each file.
    fig = plt.figure(figsize=(5, 3))

    # plt.bar(0, 0, color=color, label='KVFlow')
    # plt.text(0, 0, '0.00', ha='center', va='bottom', fontsize=12)

    for i in range(len(stat_dirs)):
        with open(os.path.join(stat_dirs[i], 'stats.pkl'), 'rb') as f:
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

        num_logical_blocks = stats['num_logical_blocks']
        num_physical_blocks = stats['num_physical_blocks']

        num_logical_blocks = num_logical_blocks[start:end]
        num_physical_blocks = num_physical_blocks[start:end]

        # Compute the average reference count.
        avg_ref_count = np.mean([x / y for x, y in zip(num_logical_blocks, num_physical_blocks)])
        saving = (1 - (1 / avg_ref_count)) * 100
        # print(f'Saving: {saving:.2f}')
        plt.bar(i, saving, color='tab:blue')
        plt.text(i, saving, f'{saving:.2f}', ha='center', va='bottom', fontsize=16)
    
    plt.xticks(range(len(stat_dirs)), [2, 4, 6], fontsize=16)
    # plt.legend(loc='upper center', fontsize=12, frameon=False, bbox_to_anchor=(0.5, 1.15), ncol=2)


def plot_parallel_gen():
    plot(stat_dirs=PARALLEL_GEN_DIRS, color='tab:blue')
    plt.xlabel('# Output sequences', fontsize=16)
    plt.ylabel('Memory saving (%)', fontsize=16)
    plt.yticks(range(0, 13, 4), fontsize=16)
    plt.ylim(bottom=0, top=12)
    plt.tight_layout()
    os.makedirs('./figures', exist_ok=True)
    plt.savefig('./figures/fig15_a.pdf')


def plot_beam():
    plot(stat_dirs=BEAM_DIRS, color='tab:blue')
    plt.xlabel('Beam width', fontsize=16)
    plt.ylabel('Memory saving (%)', fontsize=16)
    plt.ylim(bottom=0, top=70)
    plt.yticks(range(0, 71, 20), fontsize=16)
    plt.tight_layout()
    os.makedirs('./figures', exist_ok=True)
    plt.savefig('./figures/fig15_b.pdf')


if __name__ == '__main__':
    plot_parallel_gen()
    plot_beam()
