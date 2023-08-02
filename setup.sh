#!/bin/bash

set -e

# Create conda environments and install dependencies
for name in "astra" "orca" "prefix_astra" "prefix_orca";
do
    echo "Installing $name"
    # Create conda environment
    conda create -n $name python=3.9 -y
    conda activate $name

    # Install dependencies
    pip install torch==2.0.0
    pip install -r requirements.txt
    pip install git+https://github.com/huggingface/transformers@15641892985b1d77acc74c9065c332cd7c3f7d7f

    # Build and install the package
    cd $name
    echo "Installing $name"
    pip install -e .

    # Go back to the root directory
    cd ..
done
