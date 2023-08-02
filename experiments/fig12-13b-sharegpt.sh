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

# OPT-13B, ShareGPT
# system -> request rates
declare -A OPT_13B_SHAREGPT=(
    ["astra"]="1.0,1.25,1.5,1.75,1.9,2.0"
    ["orca-oracle"]="0.5,0.75,1.0,1.25,1.5"
    ["orca-power2"]="0.25,0.5,0.75,0.9"
    ["orca-constant"]="0.2,0.4,0.5,0.75"
)
# Run experiments
for system in "${SYSTEMS[@]}"; do
    python experiments/text_completion.py \
        --system $system --model opt-13b \
        --duration $DURATION --sampling n1 \
        --dataset sharegpt --rates ${OPT_13B_SHAREGPT[$system]}
done
