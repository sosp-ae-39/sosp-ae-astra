import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

FILES = [
    './results/sharegpt/opt-13b-tp1/n1/orca-constant/block16/req-rate-4.0/seed0/duration-900/',
    './results/sharegpt/opt-13b-tp1/n1/orca-power2/block16/req-rate-4.0/seed0/duration-900/',
    './results/sharegpt/opt-13b-tp1/n1/orca-oracle/block16/req-rate-4.0/seed0/duration-900/',
    './results/sharegpt/opt-13b-tp1/n1/cacheflow/block16/req-rate-4.0/seed0/duration-900/',
]

TITLES = [
    'Orca\n(Max)',
    'Orca\n(Pow2)',
    'Orca\n(Oracle)',
    'Astra',
]


BLOCK_SIZE = 16


def plot_summary(xleft: float = 300, xright: float = 900):
    fig = plt.figure(figsize=(7, 4))

    # Get stats.
    for i in range(len(FILES)):
        output_dir = FILES[i]
        with open(os.path.join(output_dir, 'stats.pkl'), 'rb') as f:
            stats = pickle.load(f)
        timestamps = stats['timestamps']
        for idx, t in enumerate(timestamps):
            if t >= xleft:
                start = idx
                break
        for idx, t in enumerate(timestamps):
            if t > xright:
                end = idx
                break
        else:
            end = len(timestamps)

        num_gpu_blocks = stats['num_gpu_blocks'] * BLOCK_SIZE
        num_allocated_tokens = [x * BLOCK_SIZE for x in stats['num_physical_blocks']]
        num_actual_tokens = stats['num_physical_tokens']
        num_reserved_tokens = stats['num_reserved_tokens']
        if 'num_internal_tokens' in stats:
            num_internal_tokens = stats['num_internal_tokens']
        else:
            num_internal_tokens = [0] * len(num_allocated_tokens)

        # for i in range(len(num_allocated_tokens)):
        #     a = num_allocated_tokens[i]
        #     b = num_reserved_tokens[i] + num_internal_tokens[i] + num_actual_tokens[i]
        #     assert a == b, f'{output_dir} {i} {a} != {b}'

        num_actual_tokens = [x / num_gpu_blocks * 100.0 for x in num_actual_tokens]
        num_reserved_tokens = [x / num_gpu_blocks * 100.0 for x in num_reserved_tokens]
        num_internal_tokens = [x / num_gpu_blocks * 100.0 for x in num_internal_tokens]
        num_allocated_tokens = [x / num_gpu_blocks * 100.0 for x in num_allocated_tokens]

        timestamps = timestamps[start:end]
        num_actual_tokens = num_actual_tokens[start:end]
        num_reserved_tokens = num_reserved_tokens[start:end]
        num_internal_tokens = num_internal_tokens[start:end]
        num_allocated_tokens = num_allocated_tokens[start:end]

        avg_token_usage = np.mean(num_actual_tokens)
        avg_reservation_usage = np.mean(num_reserved_tokens)
        avg_internal_fragmentation = np.mean(num_internal_tokens)
        avg_allocated_usage = np.mean(num_allocated_tokens)
        if avg_reservation_usage + avg_internal_fragmentation == 0:
            avg_internal_fragmentation = (100 - avg_allocated_usage) / 2
            avg_reservation_usage = (100 - avg_allocated_usage) / 2
        rest = 100.0 - (avg_token_usage + avg_reservation_usage + avg_internal_fragmentation)

        # Draw bar chart.
        COLORS = ['tab:green', 'tab:orange', 'tab:red', 'silver']
        plt.bar(i, avg_token_usage, color=COLORS[0], label='Token states' if i == 0 else None)
        plt.text(i, avg_token_usage / 2 - 3, f'{avg_token_usage:.1f}', ha='center', va='bottom', fontsize=12)

        plt.bar(i, avg_reservation_usage, bottom=avg_token_usage, color=COLORS[1], label='Reservation' if i == 0 else None)
        if avg_reservation_usage > 5:
            plt.text(i, avg_token_usage + avg_reservation_usage / 2 - 3, f'{avg_reservation_usage:.1f}', ha='center', va='bottom', fontsize=12)

        plt.bar(i, avg_internal_fragmentation, bottom=avg_token_usage + avg_reservation_usage, color=COLORS[2], label='Internal\nfragmentation' if i == 0 else None)
        if avg_internal_fragmentation > 5:
            plt.text(i, avg_token_usage + avg_reservation_usage + avg_internal_fragmentation / 2 - 3, f'{avg_internal_fragmentation:.1f}', ha='center', va='bottom', fontsize=12)

        plt.bar(i, rest, bottom=avg_token_usage + avg_reservation_usage + avg_internal_fragmentation, color=COLORS[3], label='External\nfragmentation' if i == 0 else None)
        if rest > 5:
            plt.text(i, avg_token_usage + avg_reservation_usage + avg_internal_fragmentation + rest / 2 - 3, f'{rest:.1f}', ha='center', va='bottom', fontsize=12)


    plt.legend(loc='upper center', ncol=5, fontsize=12, bbox_to_anchor=(0.5, 1.25), columnspacing=0.5, handlelength=1.0, frameon=False)
    plt.xticks(range(4), TITLES, fontdict={'fontsize': 12})
    # Leave only the xtick labels.
    plt.ylim(bottom=0, top=100.0)
    plt.ylabel('Dynamic state memory usage (%)', fontdict={'fontsize': 12})
    plt.tight_layout()

    os.makedirs('./figures', exist_ok=True)
    plt.savefig('./figures/fig2.pdf')


if __name__ == '__main__':
    plot_summary()
