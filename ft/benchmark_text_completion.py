import argparse
import logging
import numpy as np
import os
import pickle
import time
from typing import List

from tqdm import tqdm
from transformers import AutoConfig

from ft_server import FTServer, FakeFrontend, FTHandler
from trace import generate_text_completion_requests
from cacheflow.sampling_params import SamplingParams
from cacheflow.utils import get_gpu_memory, get_cpu_memory


logger = logging.getLogger(__name__)


def main(args: argparse.Namespace):
    assert args.pipeline_para_size == 1, (
        'Do not include pipeline parallelism.')

    # (num_nodes, num_devices_per_node, distributed_init_method,
    # all_stage_devices) = (
    #     initialize_ray_cluster(
    #         address='local',
    #         pipeline_parallel_size=args.pipeline_parallel_size,
    #         tensor_parallel_size=args.tensor_parallel_size))

    # Create a FT handler
    ft = FTHandler(args.model_name, args.ft_model_location,
                   args.tensor_para_size, args.pipeline_para_size,
                   args.lib_path, args.data_type, args.weights_data_type)

    # Create a FT server.
    server = FTServer(
        args.max_batch_size,
        ft,
    )

    # Create a frontend.
    frontend = FakeFrontend()

    # Generate requests.
    requests = generate_text_completion_requests(
        args.dataset,
        args.request_rate,
        args.duration,
        args.seed,
        args.n1,
        args.n2,
        args.n3,
        args.n4,
        args.n6,
        args.n2_beam,
        args.n4_beam,
        args.n6_beam,
        args.n8_beam,
    )

    # Warm up.
    logger.info('Warming up.')
    num_warmup_requests = 8
    warmup_input_len = 8
    warmup_output_len = 32
    warmup_sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=0.99,
        max_num_steps=warmup_output_len,
        use_beam_search=False,
        stop_token_ids=set(),
        num_logprobs=0,
        context_window_size=None,
    )
    for _ in range(num_warmup_requests):
        frontend._add_query([0] * warmup_input_len, warmup_sampling_params)
    server.add_sequence_groups(frontend.get_inputs())
    while True:
        server.step()
        if not server.has_unfinished_requests():
            break

    # Start benchmarking.
    logger.info('Start benchmarking.')
    # Initialize tqdm.
    pbar = tqdm(total=len(requests), desc='Finished requests')

    res = []

    finished = []
    # server.scheduler.reset_stats()
    start_time = time.time()
    while True:
        now = time.time()
        while requests:
            if requests[0][0] <= now - start_time:
                request_time, input_tokens, sampling_params = requests.pop(0)
                # print("request_time", request_time)
                # print("arrival_time", start_time + request_time)
                # print("real start time", time.time())
                frontend._add_query(
                    input_tokens, sampling_params, arrival_time=start_time + request_time)
            else:
                break
        server.add_sequence_groups(frontend.get_inputs())
        start = time.time()
        updated_seq_groups = server.step()
        if len(updated_seq_groups) > 0:
            print("time elapsed", (now - start_time) / 60, "min")
            # print("server latency", time.time() - start)

        now = time.time()
        for seq_group in updated_seq_groups:
            if not seq_group.is_finished():
                continue
            arrival_time = seq_group.arrival_time
            finish_time = now
            # print("recorded latency", finish_time - arrival_time)
            for seq in seq_group.get_seqs():
                seq_len = seq.get_len()
                output_len = seq_len - seq.prompt_len
                print("finished", len(finished), now - start, (now - arrival_time) / output_len)
                finished.append({
                    'group_id': seq_group.group_id,
                    'seq_id': seq.seq_id,
                    'arrival_time': arrival_time, 
                    'finish_time': finish_time,
                    'prompt_len': seq.prompt_len,
                    'output_len': output_len,
                })
                # res.append((finish_time - start) / output_len)
                # print(np.mean(np.array(res)))
            # pbar.update(1)

        if not (requests or server.has_unfinished_requests()):
            break
    pbar.close()

    # print(np.mean(np.array(res)))

    logger.info('Finish benchmarking. Saving stats.')
    # server.scheduler.save_stats(args.output_dir)
    with open(os.path.join(args.output_dir, 'sequences.pkl'), 'wb') as f:
        pickle.dump(finished, f)
    logger.info('Done.')


def get_sampling_dir_name(
    n1: float,
    n2: float,
    n3: float,
    n4: float,
    n6: float,
    n2_beam: float,
    n4_beam: float,
    n6_beam: float,
    n8_beam: float,
) -> str:
    method = ''
    if n1 > 0.0:
        method = 'n1' if n1 == 1.0 else method + f'n1-{n1}-'
    if n2 > 0.0:
        method = 'n2' if n2 == 1.0 else method + f'n2-{n2}-'
    if n3 > 0.0:
        method = 'n3' if n3 == 1.0 else method + f'n3-{n3}-'
    if n4 > 0.0:
        method = 'n4' if n4 == 1.0 else method + f'n4-{n4}-'
    if n6 > 0.0:
        method = 'n6' if n6 == 1.0 else method + f'n6-{n6}-'
    if n2_beam > 0.0:
        method = 'n2-beam' if n2_beam == 1.0 else method + f'n2-beam-{n2_beam}-'
    if n4_beam > 0.0:
        method = 'n4-beam' if n4_beam == 1.0 else method + f'n4-beam-{n4_beam}-'
    if n6_beam > 0.0:
        method = 'n6-beam' if n6_beam == 1.0 else method + f'n6-beam-{n6_beam}-'
    if n8_beam > 0.0:
        method = 'n8-beam' if n8_beam == 1.0 else method + f'n8-beam-{n8_beam}-'
    return method[:-1] if method.endswith('-') else method


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FT simple server.')
    # FT params
    parser.add_argument('--model-name', type=str, default='facebook/opt-125m', help='model name')
    parser.add_argument('--ft-model-location', type=str, default="opt-125m/c-model",
                        help='fastertransformer model path')
    parser.add_argument('--tensor-para-size', type=int, default=1, help='tensor parallelism size')
    parser.add_argument('--pipeline-para-size', type=int, default=1, help='pipeline parallelism size')

    parser.add_argument('--lib_path', type=str, default='fastertransformer/build/lib/libth_transformer.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp16')
    parser.add_argument('--weights_data_type', type=str, default="fp16",
                        choices=["fp32", "fp16"],
                        help='Data type of FT checkpoint weights')
    parser.add_argument('--max-batch-size', type=int, default=2560, help='maximum number of batched tokens')

    # exp params
    parser.add_argument('--output-dir', type=str, help='path to output directory', default=None)

    parser.add_argument('--dataset', type=str, help='path to dataset',
                        default="../alpaca_opt_text_completion.pkl")
    parser.add_argument('--request-rate', type=float, help='reqs/sec', default=5)
    parser.add_argument('--duration', type=int, help='duration in seconds', default=5)

    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--n1', type=float, help='ratio of requests with n=1', default=1.0)
    parser.add_argument('--n2', type=float, help='ratio of requests with n=2', default=0.0)
    parser.add_argument('--n3', type=float, help='ratio of requests with n=3', default=0.0)
    parser.add_argument('--n4', type=float, help='ratio of requests with n=4', default=0.0)
    parser.add_argument('--n6', type=float, help='ratio of requests with n=6', default=0.0)
    parser.add_argument('--n2-beam', type=float, help='ratio of requests with n=2 & beam search', default=0.0)
    parser.add_argument('--n4-beam', type=float, help='ratio of requests with n=4 & beam search', default=0.0)
    parser.add_argument('--n6-beam', type=float, help='ratio of requests with n=6 & beam search', default=0.0)
    parser.add_argument('--n8-beam', type=float, help='ratio of requests with n=8 & beam search', default=0.0)
    args = parser.parse_args()
    if args.n1 + args.n2 + args.n3 + args.n4 + args.n6 + args.n2_beam + args.n4_beam + args.n6_beam + args.n8_beam != 1.0:
        raise ValueError('The ratios of requests must sum to 1.')

    model_name = args.model_name
    dataset_name = 'sharegpt' if 'sharegpt' in args.dataset else 'alpaca'
    if 'opt' in model_name:
        if 'opt' not in args.dataset.lower():
            raise ValueError(f'OPT models can only be used with OPT datasets.')
    elif 'llama' in model_name:
        if 'llama' not in args.dataset.lower():
            raise ValueError(f'Llama models can only be used with Llama datasets.')

    sample_dir = get_sampling_dir_name(
        args.n1, args.n2, args.n3, args.n4, args.n6, args.n2_beam, args.n4_beam, args.n6_beam, args.n8_beam)
    if args.output_dir is None:
        args.output_dir = os.path.join(
            'exp',
            dataset_name,
            f'{model_name}-tp{args.tensor_para_size}-pp{args.pipeline_para_size}',
            sample_dir,
            'ft',
            f'bs{args.max_batch_size}',
            f'req-rate-{args.request_rate}',
            f'seed{args.seed}',
            f'duration-{args.duration}',
        )
    os.makedirs(args.output_dir, exist_ok=True)

    # Set up logging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(args.output_dir, 'log.txt')),
        ],
    )
    logger.info(args)

    main(args)
