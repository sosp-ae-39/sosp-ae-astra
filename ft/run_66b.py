import os

# T4:
# 1.3b: bs 8 for sharegpt
# 2.7b: bs 2 for sharegpt

# A100 40G:
# 13b: 16 for alpaca, 6 for sharegpt
# 13b n2-beam: 14 for alpaca

# 8 * A100 80G:
# 175b: 88 for alpaca

# 4 * A100 40G:
# 66b: 8 for alpaca, 2 for sharegpt


alpaca = "../alpaca_opt_text_completion.pkl"
sharegpt = "../sharegpt_clean_opt_text_completion.pkl"

rates = {
         alpaca: [0.1, 0.5, 1, 1.25, 1.5, 1.75, 2, 2.5],
         sharegpt: [0.01, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2],
        }

bs = {alpaca: 8,
      sharegpt: 2}

for dataset in [alpaca, sharegpt]:
    for rate in rates[dataset]:
        model_name = "facebook/opt-66b"
        ft_model_location = "opt-66b/c-model"
        duration = 3600
        max_bs = bs[dataset]
        num_gpu = 4
    
        cmd = f"mpirun --allow-run-as-root -n {num_gpu} " \
              f"python benchmark_text_completion.py " \
              f"--dataset {dataset} " \
              f"--request-rate {rate} " \
              f"--model-name {model_name} --ft-model-location {ft_model_location} " \
              f"--duration {duration} " \
              f"--max-batch-size {max_bs} " \
              f"--data_type fp16 " \
              f"--weights_data_type fp16 " \
              f"--tensor-para-size {num_gpu} "
        os.system(cmd)
