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

# NOTE: Each experiment takes 4 GPUs.

# OPT-66B, ShareGPT
declare -A OPT_66B_SHAREGPT=(
    ["astra"]="0.4,0.6,0.8,0.9,1.0"
    ["orca-oracle"]="0.2,0.3,0.4,0.5"
    ["orca-power2"]="0.1,0.2,0.25,0.35"
    ["orca-constant"]="0.1,0.2,0.3"
)
for system in "${SYSTEMS[@]}"; do
    python experiments/text_completion.py \
        --system $system --model opt-66b \
        --duration $DURATION --sampling n1 \
        --dataset sharegpt --rates ${OPT_66B_SHAREGPT[$system]}
done

# OPT-66B, Alpaca
declare -A OPT_66B_ALPACA=(
    ["astra"]="11.0,13.0,15.0,17.5,18.5,19.0,20.0"
    ["orca-oracle"]="6.0,8.0,10.0,11.0,12.0"
    ["orca-power2"]="3.0,5.0,6.0,7.0"
    ["orca-constant"]="0.05,0.1,0.15,0.25"
)
for system in "${SYSTEMS[@]}"; do
    python experiments/text_completion.py \
        --system $system --model opt-66b \
        --duration $DURATION --sampling n1 \
        --dataset alpaca --rates ${OPT_66B_ALPACA[$system]}
done
