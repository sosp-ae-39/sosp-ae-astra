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

# Beam search, beam=2
declare -A BEAM2=(
    ["astra"]="10.0,12.5,15.0,16.0,17.5,19.0"
    ["orca-oracle"]="6.0,8.0,9.0,10.0,11.0"
    ["orca-power2"]="3.0,5.0,6.0,7.0,8.0,9.0"
    ["orca-constant"]="0.5,1.0,1.25,1.5,2.0"
)
for system in "${SYSTEMS[@]}"; do
    python experiments/text_completion.py \
        --system $system --model opt-13b \
        --duration $DURATION --sampling n2-beam \
        --dataset alpaca --rates ${BEAM2[$system]}
done

# Beam search, beam=4
declare -A BEAM4=(
    ["astra"]="6.0,8.0,9.0,10.0,11.0"
    ["orca-oracle"]="3.0,4.0,4.5,5.0,6.0"
    ["orca-power2"]="1.0,2.0,3.0,3.5,4.0,4.5"
    ["orca-constant"]="0.1,0.25,0.5,0.75,1.0"
)
for system in "${SYSTEMS[@]}"; do
    python experiments/text_completion.py \
        --system $system --model opt-13b \
        --duration $DURATION --sampling n4-beam \
        --dataset alpaca --rates ${BEAM4[$system]}
done

# Beam search, beam=6
declare -A BEAM6=(
    ["astra"]="4.0,6.0,6.5,7.0,8.0,8.5"
    ["orca-oracle"]="1.0,2.0,3.0,3.5,4.0,4.5"
    ["orca-power2"]="0.5,1.5,2.0,2.5,3.0"
    ["orca-constant"]="0.1,0.25,0.5,0.75,1.0"
)
for system in "${SYSTEMS[@]}"; do
    python experiments/text_completion.py \
        --system $system --model opt-13b \
        --duration $DURATION --sampling n6-beam \
        --dataset alpaca --rates ${BEAM6[$system]}
done
