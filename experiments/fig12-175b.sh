#/bin/bash

set -e

# NOTE: The duration of each experiment is 3600 sec in the paper.
# However, for the sake of time, we set it to 600 sec here.
DURATION=600

SYSTEMS=(
    "astra"
    "orca-oracle"
    "orca-power2"
    "orca-constant"
)

# NOTE: Each experiment takes 8 GPUs.

# OPT-175B, ShareGPT
declare -A OPT_175B_SHAREGPT=(
    ["astra"]="1.5,1.75,2.0,2.25,2.5,3.0"
    ["orca-oracle"]="0.8,1.0,1.2,1.4,1.6,1.8"
    ["orca-power2"]="0.5,0.75,1.0,1.25,1.5"
    ["orca-constant"]="0.25,0.5,0.75,1.0.1.25"
)
for system in "${SYSTEMS[@]}"; do
    python experiments/text_completion.py \
        --system $system --model opt-175b \
        --duration $DURATION --sampling n1 \
        --dataset sharegpt --rates ${OPT_175B_SHAREGPT[$system]}
done

# OPT-175B, Alpaca
declare -A OPT_175B_ALPACA=(
    ["astra"]="15.0,17.5,19.0,20.0,22.0"
    ["orca-oracle"]="15.0,17.5,19.0,20.0,22.0"
    ["orca-power2"]="10.0,15.0,16.0,17.0,19.0"
    ["orca-constant"]="1.25,2.5,4.0,5.0,7.0"
)
for system in "${SYSTEMS[@]}"; do
    python experiments/text_completion.py \
        --system $system --model opt-175b \
        --duration $DURATION --sampling n1 \
        --dataset alpaca --rates ${OPT_175B_ALPACA[$system]}
done
