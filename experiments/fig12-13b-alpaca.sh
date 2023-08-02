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

# OPT-13B, Alpaca
declare -A OPT_13B_ALPACA=(
    ["astra"]="20.0,22.5,25.0,27.5,29.0,31.5"
    ["orca-oracle"]="15.0,17.5,20.0,21.0,22.5,24.0"
    ["orca-power2"]="7.5,10.0,12.5,15.0,16.0,17.0,18.0"
    ["orca-constant"]="2.0,3.0,4.0,5.0"
)
for system in "${SYSTEMS[@]}"; do
    python experiments/text_completion.py \
        --system $system --model opt-13b \
        --duration $DURATION --sampling n1 \
        --dataset alpaca --rates ${OPT_13B_ALPACA[$system]}
done
