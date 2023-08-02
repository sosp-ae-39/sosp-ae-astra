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

# Parallel sampling, n=2
declare -A N2=(
    ["astra"]="12.5,15.0,18.0,19.0,20.0"
    ["orca-oracle"]="10.0,11.0,12.0,13.0,14.0"
    ["orca-power2"]="6.0,7.0,8.0,9.0,10.0,12.0"
    ["orca-constant"]="1.0,1.5,2.0,2.5"
)
for system in "${SYSTEMS[@]}"; do
    python experiments/text_completion.py \
        --system $system --model opt-13b \
        --duration $DURATION --sampling n2 \
        --dataset alpaca --rates ${N2[$system]}
done

# Parallel sampling, n=4
declare -A N4=(
    ["astra"]="6.0,8.0,10.0,11.0,12.0"
    ["orca-oracle"]="4.0,5.0,6.0,7.0,8.0,9.0"
    ["orca-power2"]="2.0,3.0,4.0,5.0,6.0"
    ["orca-constant"]="0.1,0.25,0.5,0.75,1.0"
)
for system in "${SYSTEMS[@]}"; do
    python experiments/text_completion.py \
        --system $system --model opt-13b \
        --duration $DURATION --sampling n4 \
        --dataset alpaca --rates ${N4[$system]}
done

# Parallel sampling, n=6
declare -A N6=(
    ["astra"]="4.0,5.0,6.0,7.0,8.0,9.0"
    ["orca-oracle"]="2.0,3.0,4.0,5.0,6.0"
    ["orca-power2"]="1.0,2.0,2.5,3.0,4.0,4.5"
    ["orca-constant"]="0.1,0.25,0.5,0.75,1.0"
)
for system in "${SYSTEMS[@]}"; do
    python experiments/text_completion.py \
        --system $system --model opt-13b \
        --duration $DURATION --sampling n6 \
        --dataset alpaca --rates ${N6[$system]}
done
