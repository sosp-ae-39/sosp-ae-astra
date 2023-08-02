import argparse
import os


def run_cmd(cmd):
    print(cmd)
    ret = os.system(cmd)
    if ret != 0:
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, required=True)
    parser.add_argument('--len-estimator', type=str, choices=['oracle', 'power2', 'constant'], required=True)
    args = parser.parse_args()

    rates = [0.10, 0.20, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    new_prob = 0.1
    duration = args.duration
    len_estimator = args.len_estimator
    for rate in rates:
        cmd = ""
        cmd += f"conda activate orca;"
        cmd += f"cd ./orca;"
        cmd += f"python benchmark/benchmark_chatbot.py --len-estimator {len_estimator} --dataset ../datasets/sharegpt_clean_lang_10k_opt_tokenized.pkl --model facebook/opt-13b --request-rate {rate} --duration {duration} --n1 1.0 --new-prob {new_prob} --use-dummy"
        cmd = f'bash -i -c "{cmd}"'
        run_cmd(cmd)
