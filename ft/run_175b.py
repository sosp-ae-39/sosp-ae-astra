import os

# T4:
# 1.3b: bs 8 for sharegpt
# 2.7b: bs 2 for sharegpt

# A100 40G:
# 13b: 16 for alpaca, 6 for sharegpt
# 13b n2-beam: 14 for alpaca

# 8 * A100 80G:
# 175b: 80 for alpaca, 24 for sharegpt

alpaca = "../alpaca_opt_text_completion.pkl"
sharegpt = "../sharegpt_clean_opt_text_completion.pkl"

rates = {
         alpaca: [0.1, 0.5, 1, 1.5, 2, 5],
         sharegpt: [0.01, 0.05, 0.1, 0.15, 0.2, 0.3],
        }

bs = {alpaca: 80,
      sharegpt: 24}

for dataset in [alpaca, sharegpt]:
    # 5 will cause explosion
    for rate in rates[dataset]:
        model_name = "facebook/opt-175b"
        ft_model_location = "opt-175b/c-model"
        duration = 3600
        if dataset == alpaca and rates in [1.5, 2, 5]:
            duration = 600
        max_bs = bs[dataset]
        num_gpu = 8
    
        cmd = f"mpirun --allow-run-as-root -n {num_gpu} " \
              f"python benchmark_text_completion.py --request-rate {rate} " \
              f"--model-name {model_name} --ft-model-location {ft_model_location} " \
              f"--duration {duration} " \
              f"--max-batch-size {max_bs} " \
              f"--data_type fp16 " \
              f"--weights_data_type fp16 " \
              f"--tensor-para-size {num_gpu} " \
              f"--dataset {dataset} "
        os.system(cmd)
