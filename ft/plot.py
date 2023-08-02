import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="alpaca")
parser.add_argument("--model-name", type=str, default="facebook/opt-13b")
parser.add_argument("--tensor-para-size", type=str, default=1)
parser.add_argument('--n1', type=float, help='ratio of requests with n=1', default=1.0)
parser.add_argument('--n2-beam', type=float, help='ratio of requests with n=2 & beam search', default=0.0)
parser.add_argument('--max-batch-size', type=int, default=2560)
parser.add_argument('--duration', type=int, default=3600)
args = parser.parse_args()

DATASET = args.dataset
MODEL = args.model_name.split("/")[-1]
MODEL_PATH = args.model_name + f"-tp{args.tensor_para_size}-pp1"
MAX_BATCH_SIZE = args.max_batch_size
DURATION = args.duration
WARMUP = DURATION * 60
RATIO = "n1" if args.n1 == 1 else "none"

alpaca_schemes = {
    'ft': f'FasterTransformer (Max, bs={MAX_BATCH_SIZE})',
}
schemes = alpaca_schemes

# Create a figure
fig = plt.figure(figsize=(10, 6))

for scheme in schemes.keys():
    save_dir = os.path.join(
        "exp",
        DATASET,
        MODEL_PATH,
        RATIO,
        scheme,
        f'bs{MAX_BATCH_SIZE}',
        # 'no_beam',
    )
    results = os.listdir(save_dir)

    request_rate_to_normalized_latency = {}
    for result in results:
        request_rate = float(result[len("req-rate-"):])

        seed_dir = os.path.join(save_dir, result)
        seeds = os.listdir(seed_dir)
        # if len(seeds) < 3:
        #     continue
        seeds = ['seed0']

        sum = 0
        for i, seed in enumerate(seeds):
            d = os.path.join(seed_dir, seed, f'duration-{DURATION}')
            filename = os.path.join(d, 'sequences.pkl')
            if not os.path.exists(filename):
                print(f"{filename} does not exist!")
                continue
            with open(os.path.join(d, 'sequences.pkl'), 'rb') as f:
                data = pickle.load(f)
            print("rate:", request_rate, "num requests", len(data))
            first_arrival = min([x["arrival_time"] for x in data])
            last_finish = max([x["finish_time"] for x in data])
            exp_duration = last_finish - first_arrival

            normalized_latency = []
            ratio = 10
            for req in data:
                if req["arrival_time"] < first_arrival + exp_duration // ratio:
                    continue
                # if req["arrival_time"] > first_arrival + DURATION - exp_duration // ratio:
                #     continue
                latency = req["finish_time"] - req["arrival_time"]
                output_len = req["output_len"]
                normalized_latency.append(latency / output_len)
            print(len(normalized_latency))
            sum += np.mean(normalized_latency)
        if sum == 0: continue
        request_rate_to_normalized_latency[request_rate] = sum / len(seeds)

    request_rates = sorted(request_rate_to_normalized_latency.keys())
    normalized_latencies = [request_rate_to_normalized_latency[request_rate] for request_rate in request_rates]
    print(scheme, request_rates, normalized_latencies)
    plt.plot(request_rates, normalized_latencies, label=schemes[scheme], marker='o')

plt.ylim(bottom=0, top=1)
# plt.yscale('log')
plt.subplots_adjust(right=0.6)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.title(f'{DATASET} dataset ({MODEL}, A100-40G)')
plt.xlabel('Request rate (reqs/s)')
plt.ylabel('Normalized latency (s/token)')

plt.savefig(f'{DATASET}_{MODEL}.png')
