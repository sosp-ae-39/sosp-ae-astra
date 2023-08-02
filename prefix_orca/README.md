# Reproduce Orca Prefix Experiments (Sec 6.4, Figure 16, Orca (Oracle))

## Installation

```bash
pip install psutil numpy ray torch
pip install git+https://github.com/huggingface/transformers  # Required for LLaMA.
pip install sentencepiece  # Required for LlamaTokenizer.
pip install ninja  # To parallelize the compilation of flash-attn.
pip install flash-attn  # This may take up to 10 mins.
pip install -e .
```

## Load LLaMA weights

This experiment requires LLaMA-13b weights with Huggingface format, under `~/hf-llama/llama-13b/`.

Since LLaMA weight is not fully public, we cannot directly download the LLaMA weights from huggingface. Therefore, you need to follow the following process to load the LLaMA weights.

1. Converting LLaMA weights to huggingface format with [this script](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py).
    ```bash
    python src/transformers/models/llama/convert_llama_weights_to_hf.py \
        --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path/llama-7b
    ```
    Please make sure that `llama` is included in the output directory name.
2. For all the commands above, specify the model with `--model /output/path/llama-7b` to load the model. For example:
    ```bash
    python simple_server.py --model /output/path/llama-7b
    python -m cacheflow.http_frontend.fastapi_frontend --model /output/path/llama-7b
    ```

## Run experiments

Under the current directory, run:

```bash
./benchmark_prefix_translation_orca.sh
```
