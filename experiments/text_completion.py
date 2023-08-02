
import argparse
import subprocess
from typing import Tuple, Optional


def get_model(model_name: str) -> Tuple[str, int]:
    OPT_MODELS = [
        'opt-13b',
        'opt-66b',
        'opt-175b',
    ]
    assert model_name in OPT_MODELS
    if model_name == 'opt-13b':
        return 'facebook/opt-13b', 1
    elif model_name == 'opt-66b':
        return 'facebook/opt-66b', 4
    elif model_name == 'opt-175b':
        return '../opt-175b', 8
    else:
        raise ValueError(f'Unknown model: {model_name}')


def get_dataset(dataset_name, model_name) -> str:
    assert 'opt' in model_name
    if dataset_name == 'alpaca':
        return "../datasets/alpaca_opt_text_completion.pkl"
    else:
        assert dataset_name == 'sharegpt'
        return "../datasets/sharegpt_opt_text_completion_length.pkl"


def run_exp(
    system: str,
    dataset: str,
    model: str,
    block_size: int,
    tp: int,
    samping: str,
    request_rate: float,
    duration: int,
    seed: int,
    timeout: Optional[int],
    do_memory_analysis: bool,
    always_swap: bool,
    memory_util: Optional[float],
) -> None:
    if 'orca' in system:
        conda_env = 'orca'
        exp_path = './orca'
        len_estimator = system[len('orca-'):]
    else:
        assert system == 'astra'
        conda_env = 'astra'
        exp_path = './astra'
        len_estimator = None

    cmd = (f'conda activate {conda_env}; '
           f'cd {exp_path}; '
           'python benchmark/benchmark_text_completion.py '
           f'--dataset {dataset} --model {model} -tp {tp} '
           f'--block-size {block_size} '
           f'--{samping} 1.0 --request-rate {request_rate} '
           f'--duration {duration} --seed {seed}')
    if len_estimator is not None:
        cmd += f' --len-estimator {len_estimator}'
    if  'opt-175b' in model:
        cmd += ' --use-dummy-weights'
    if timeout is not None:
        cmd += f' --timeout {timeout}'
    if do_memory_analysis:
        cmd += ' --do-memory-analysis'
    if always_swap:
        cmd += ' --always-swap'
    if memory_util is not None:
        cmd += f' --gpu-memory-utilization {memory_util}'
    # Run cmd under conda env
    cmd = f'bash -i -c "{cmd}"'
    subprocess.run(cmd, shell=True, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--system', type=str, choices=['astra', 'orca-oracle', 'orca-power2', 'orca-constant'])
    parser.add_argument('--model', type=str, choices=['opt-13b', 'opt-66b', 'opt-175b'])
    parser.add_argument('--block-size', type=int, default=16)
    parser.add_argument('--dataset', type=str, choices=['alpaca', 'sharegpt'])
    parser.add_argument('--sampling', type=str, choices=['n1', 'n2', 'n3', 'n4', 'n6', 'n2-beam', 'n4-beam', 'n6-beam', 'n8-beam'])
    parser.add_argument('--rates', type=str, required=True)
    parser.add_argument('--duration', type=int, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--timeout', type=int, default=None)
    parser.add_argument('--do-memory-analysis', action='store_true')
    parser.add_argument('--always-swap', action='store_true')
    parser.add_argument('--memory-util', type=float, default=None)

    args = parser.parse_args()
    model, tp = get_model(args.model)
    dataset = get_dataset(args.dataset, args.model)
    request_rates = [float(rate) for rate in args.rates.split(',')]
    for request_rate in request_rates:
        run_exp(
            args.system,
            dataset,
            model,
            args.block_size,
            tp,
            args.sampling,
            request_rate,
            args.duration,
            args.seed,
            args.timeout,
            args.do_memory_analysis,
            args.always_swap,
            args.memory_util,
        )
