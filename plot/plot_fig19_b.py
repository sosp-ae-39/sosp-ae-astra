import argparse
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


BLOCK_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

def get_results(save_dir: str) -> List[Dict[str, Any]]:
    with open(os.path.join(save_dir, 'sequences.pkl'), 'rb') as f:
        results = pickle.load(f)
    return results


def get_request_rate(save_dir: str) -> float:
    """Get request rate from save_dir name."""
    # Directory name format:
    # .../req-rate-{req_rate}/seed-{seed}/duration-{duration}
    save_dir = os.path.abspath(save_dir)
    dir_names = save_dir.split('/')

    request_rate = None
    for dir_name in dir_names:
        if 'req-rate-' in dir_name:
            if request_rate is not None:
                raise ValueError(f'Found multiple request rates in {save_dir}')
            request_rate = float(dir_name.split('-')[-1])
    if request_rate is None:
        raise ValueError(f'Cannot find request rate in {save_dir}')
    return request_rate


def get_model(save_dir: str) -> Tuple[str, int]:
    save_dir = os.path.abspath(save_dir)
    dir_names = save_dir.split('/')

    model = None
    for dir_name in dir_names:
        if '-tp' in dir_name:
            if model is not None:
                raise ValueError(f'Found multiple models in {save_dir}')
            model = dir_name.split('-tp')[0]
            tp = int(dir_name.split('-tp')[-1])
    if model is None:
        raise ValueError(f'Cannot find model in {save_dir}')
    return model, tp


def get_system(save_dir: str) -> str:
    save_dir = os.path.abspath(save_dir)
    dir_names = save_dir.split('/')

    for dir_name in dir_names:
        if dir_name.startswith('orca-'):
            return dir_name
        if dir_name == 'cacheflow':
            return dir_name
    raise ValueError(f'Cannot find system in {save_dir}')


def get_sampling(save_dir: str) -> str:
    save_dir = os.path.abspath(save_dir)
    dir_names = save_dir.split('/')

    for dir_name in dir_names:
        if dir_name.startswith('n'):
            if dir_name.endswith('-beam'):
                return dir_name
            if dir_name[1:].isdigit():
                return dir_name
    raise ValueError(f'Cannot find sampling method in {save_dir}')


def get_block_size(save_dir: str) -> int:
    save_dir = os.path.abspath(save_dir)
    dir_names = save_dir.split('/')

    for dir_name in dir_names:
        if dir_name == 'block':
            continue
        if dir_name.startswith('block') and len(dir_name) > 5 and 'req-rate' not in dir_name:
            return int(dir_name[5:])
    raise ValueError(f'Cannot find block size in {save_dir}')


def plot_normalized_latency(
    exp_dir: str,
    request_rate: float,
    duration: int,
    seed: int,
    warmup: int,
    xlim: Optional[float],
    ylim: Optional[float],
    log_scale: bool,
    format: str,
) -> None:
    # Get leaf directories.
    save_dirs = []
    for root, dirs, files in os.walk(exp_dir):
        if dirs:
            continue
        if 'sequences.pkl' not in files:
            continue
        if f'seed{seed}' not in root:
            continue
        if f'duration-{duration}' not in root:
            continue
        if 'cacheflow' not in root:
            continue
        save_dirs.append(root)

    # Plot normalized latency.
    recompte = {}
    swap = {}
    for save_dir in save_dirs:
        per_seq_norm_latencies = []
        results = get_results(save_dir)
        for seq in results:
            arrival_time = seq['arrival_time']
            finish_time = seq['finish_time']
            output_len = seq['output_len']
            if arrival_time < warmup:
                continue
            latency = finish_time - arrival_time
            norm_latency = latency / output_len
            per_seq_norm_latencies.append(norm_latency)

        normalized_latency = np.mean(per_seq_norm_latencies)
        block_size = get_block_size(save_dir)

        if 'cacheflow-swap' in save_dir:
            perf_per_size = swap
        else:
            perf_per_size = recompte

        if block_size not in perf_per_size:
            perf_per_size[block_size] = ([], [])
        perf_per_size[block_size][0].append(get_request_rate(save_dir))
        perf_per_size[block_size][1].append(normalized_latency)

    # Plot normalized latency.
    plt.figure(figsize=(4, 3))
    latencies = []
    for block_size in BLOCK_SIZES:
        # Sort by request rate.
        r, l = recompte[block_size]
        r, l = zip(*sorted(zip(r, l)))
        for x in zip(r, l):
            if x[0] == request_rate:
                latencies.append(x[1])
                break
        else:
            raise ValueError(f'Cannot find latency for block size {block_size}')
    plt.plot(BLOCK_SIZES, latencies, marker='o', markersize=4, linewidth=1, color='tab:blue', label='Recompute')

    latencies = []
    for block_size in BLOCK_SIZES:
        # Sort by request rate.
        r, l = swap[block_size]
        r, l = zip(*sorted(zip(r, l)))
        for x in zip(r, l):
            if x[0] == request_rate:
                latencies.append(x[1])
                break
        else:
            raise ValueError(f'Cannot find latency for block size {block_size}')
    plt.plot(BLOCK_SIZES, latencies, marker='o', markersize=4, linewidth=1, color='tab:red', label='Swap')

    plt.xlabel('Block size', fontsize=12)
    plt.xscale('log', base=2)
    plt.xticks([1, 2, 4, 8, 16, 32, 64, 128, 256], labels=['1', '2', '4', '8', '16', '32', '64', '128', '256'])
    plt.ylabel('Normalized latency (s/token)', fontsize=12)

    if log_scale:
        plt.yscale('log')
    if xlim is not None:
        plt.xlim(left=0, right=xlim)
    if ylim is not None:
        if log_scale:
            plt.ylim(top=ylim)
        else:
            plt.ylim(bottom=0, top=ylim)
    plt.legend(fontsize=12, frameon=False, bbox_to_anchor=(1.0, 1.0), loc='upper right', borderpad=0.0)
    plt.tight_layout()

    # Save figure.
    figname='fig19_b.pdf'
    os.makedirs('./figures', exist_ok=True)
    plt.savefig(os.path.join('figures', figname), bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default='./results/sharegpt/opt-13b-tp1/n1/')
    parser.add_argument('--req', type=float, default=2.01)
    parser.add_argument('--duration', type=int, default=600)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmup', type=int, default=300)
    parser.add_argument('--xlim', type=float, required=False, default=None)
    parser.add_argument('--ylim', type=float, required=False, default=None)
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--format', choices=['png', 'pdf'], default='pdf')
    args = parser.parse_args()

    plot_normalized_latency(
        args.exp_dir, args.req, args.duration, args.seed, args.warmup, args.xlim, args.ylim, args.log, args.format)
