#!/bin/bash

# Script to prepare data weights using SyneTune venv

# Source the SyneTune virtual environment
source /leonardo_work/AIFAC_L01_028/hmahadik/venv_synetune_experiments/bin/activate

# Change to the synetune-experiments directory
cd /leonardo_work/AIFAC_L01_028/hmahadik/synetune-experiments

# Run sample_data_weights.py and capture output
OUTPUT=$(python sample_data_weights.py 2>&1)

# Parse the output and write to file
echo "$OUTPUT" | awk '/New data weights \(sum to 100\):/ {flag=1; next} flag && /^\/.*: [0-9]/ {split($0, a, ": "); print a[2], a[1]}' > ${BASE_PATH}/run11/data_paths.txt
