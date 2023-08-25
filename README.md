# sosp2023-ae

**Note:** This repo is merely for reproducing the experiments in our SOSP 2023 paper submission. If you are looking into testing the performance of current vLLM and PagedAttention, please refer to the [vLLM repo](https://github.com/vllm-project/vllm).

**Note:** The original Orca paper is close sourced. The Orca implementation in this repo is based on the vLLM architecture, and may not be exactly the same as the Orca implementation in the original paper. The results for the Orca in this repo should be understood as a proxy for the performance of the original Orca system.

## Setting Up the Environment

1. Install Anaconda

Anaconda can be downloaded by:
```bash
wget https://repo.anaconda.com/archive/Anaconda3-2023.07-1-Linux-x86_64.sh
```

2. Set up new conda environments by running `setup.sh`:

```bash
bash setup.sh
```
This will create 4 new conda environments: `astra`, `orca`, `prefix-astra`, and `prefix-orca`.
This process can take up to 1 hour due to CUDA compilation.

## Reproducing Main Experiments & Figures

As running all experiments takes 200+ A100 GPU-hours and may require multi-GPU instances, we highlight a few main experiments in the paper and leave the others optional for the evaluators. Additionally, we shrink the trace length from 1hr to 10min for faster experiments.

### Figure 12

Single GPU (1xA100-40GB) experiment with OPT-13B
```bash
# NOTE: This experiment may take 6+ hours.
bash experiments/fig12-13b-sharegpt.sh
python plot/plot_fig12.py results --subset n1-sharegpt --duration 600  # Saved to figures/n1-sharegpt.pdf

# Optional: Same experiment on the Alpaca dataset.
# NOTE: This experiment may take 6+ hours.
bash experiments/fig12-13b-alpaca.sh
python plot/plot_fig12.py results --subset n1-alpaca --duration 600  # Saved to figures/n1-alpaca.pdf
```

### Figure 14
```bash
# NOTE: This experiment may take 18+ hours.
bash experiments/fig14-beam.sh
python plot/plot_fig14.py results --subset beam --duration 600  # Saved to figures/beam.pdf

# Optional: Experiment with parallel sampling.
# NOTE: This experiment may take 18+ hours.
bash experiments/fig14-parallel.sh
python plot/plot_fig14.py results --subset parallel --duration 600  # Saved to figures/parallel.pdf
```

## Reproducing Other Experiments & Figures

### Figure 2
```bash
# NOTE: This experiment may take 6+ hours.
bash experiments/fig2.sh
python plot/plot_fig2.py  # Saved to figures/fig2.pdf
```

### Figure 12

- Multi-GPU (4xA100-40GB) experiment with OPT-66B
```bash
# NOTE: This experiment may take 12+ hours.
bash experiments/fig12-66b.sh
python plot/plot_fig12.py results --subset n1-sharegpt --duration 600  # Saved to figures/n1-sharegpt.pdf
python plot/plot_fig12.py results --subset n1-alpaca --duration 600  # Saved to figures/n1-alpaca.pdf
```

- Multi-GPU (8xA100-80GB) experiment with OPT-175B
```bash
# NOTE: This experiment may take 12+ hours.
bash experiments/fig12-175b.sh
python plot/plot_fig12.py results --subset n1-sharegpt --duration 600  # Saved to figures/n1-sharegpt.pdf
python plot/plot_fig12.py results --subset n1-alpaca --duration 600  # Saved to figures/n1-alpaca.pdf
```

- FasterTransformer baseline

1. Create an instance for the machine image `sosp-ae-ft` under the `Alpa-BAIR` GCP project.
   1x A100 (40GB) for OPT-13B
   4x A100 (40GB) for OPT-66B
   8x A100 (40GB) for OPT-175B

2. Launch docker
```bash
sudo bash ~/cacheflow/benchmark/ft/fastertransformer/launch_docker.sh
```

3. Run experiments
```bash
# NOTE: This experiment may take 20+ hours.
python ~/cacheflow/benchmark/ft/run_13b.py

# NOTE: This experiment may take 20+ hours.
python ~/cacheflow/benchmark/ft/run_66b.py

# NOTE: This experiment may take 20+ hours.
python ~/cacheflow/benchmark/ft/run_175b.py
```
A direct run may take 20+ hours for each command, but one can split the cases and run them in parallel (see the script).
The results will be stored in `~/cacheflow/benchmark/ft/exp`, which will be gathered with results for other systems to generate the plots.

### Figure 13
```bash
# NOTE: This experiment may take 1.5+ hours.
bash experiments/fig13.sh
python plot/plot_fig13.py  # Saved to figures/fig13_a.pdf and figures/fig13_b.pdf
```

### Figure 15
```bash
# NOTE: This experiment may take 3+ hours.
bash experiments/fig15.sh
python plot/plot_fig15.py  # Saved to figures/fig15_a.pdf and figures/fig15_b.pdf
```

### Figure 16
Execute these commands in the root directory of this repo (`~/sosp2023-ae` by default).

```bash
conda activate prefix_astra
pushd prefix_astra
# Run the experiments for Astra (~7 hrs)
bash -x benchmark_prefix_translation_astra.sh
popd
```

```bash
conda activate prefix_orca
pushd prefix_orca
# Run the experiments for Orca (~6 hrs)
bash -x benchmark_prefix_translation_orca.sh
popd
```

After these scripts completes, you will see `prefix_exp` directory under the root directory of this repo.

Then execute this command to plot the figure:

```bash
python plot/plot_fig16.py prefix_exp/
```

The figure will be saved as `figures/prefix.pdf`. This figure corresponds to Figure 16 in the paper.


### Figure 17
```bash
# NOTE: This experiment may take 12+ hours.
bash experiments/fig17.sh
python plot/plot_fig17.py results # Saved to figures/chat-sharegpt.pdf
```

### Figure 18
```bash
# NOTE: This experiment may take 2+ hours.
bash experiments/fig18-b.sh
python plot/plot_fig18_b.py  # Saved to figures/fig18_b.pdf
```

### Figure 19
```bash
# NOTE: This experiment may take 10+ minutes.
bash experiments/fig19-a.sh
python plot/plot_fig19_a.py  # Saved to figures/fig19_a.pdf

# NOTE: This experiment may take 2+ hours.
bash experiments/fig19-b.sh
python plot/plot_fig19_b.py  # Saved to figures/fig19_b.pdf
```
