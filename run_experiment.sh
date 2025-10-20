#!/bin/bash

# Exit on any error
set -e

# --- Configuration ---
POISON_RATES=(0.0 0.0001 0.0005 0.001 0.005 0.01 0.05)
DATA_DIR="data"
ADAPTERS_DIR="adapters"
RESULTS_DIR="results"

# --- Create Directories ---
mkdir -p $DATA_DIR
mkdir -p $ADAPTERS_DIR
mkdir -p $RESULTS_DIR

# --- Step 1: Prepare all datasets ---
echo "--- Preparing Datasets ---"
python prepare_data.py
echo "--- Dataset preparation complete. ---"
echo ""

# Clear previous summary log
> $RESULTS_DIR/summary.log

# --- Step 2 & 3: Loop, Fine-Tune, and Evaluate ---
for rate in "${POISON_RATES[@]}"; do
    echo "======================================================"
    echo "Processing Poison Rate: $rate"
    echo "======================================================"

    DATASET_FILE="$DATA_DIR/sft_data_rho_${rate}.jsonl"
    ADAPTER_PATH="$ADAPTERS_DIR/gemma_2b_it_lora_rho_${rate}"
    RESULTS_FILE="$RESULTS_DIR/results_rho_${rate}.json"

    # --- Fine-Tune ---
    echo "--- Starting Fine-Tuning for rate $rate ---"
    python finetune.py \
        --dataset_path "$DATASET_FILE" \
        --output_dir "$ADAPTER_PATH"
    echo "--- Fine-Tuning for rate $rate complete. ---"
    echo ""

    # --- Evaluate ---
    echo "--- Starting Evaluation for rate $rate ---"
    python evaluate.py \
        --adapter_path "$ADAPTER_PATH" \
        --results_file "$RESULTS_FILE"
    echo "--- Evaluation for rate $rate complete. ---"
    echo ""
done

echo "******************************************************"
echo "All experiments complete. Summary log:"
cat $RESULTS_DIR/summary.log
echo "******************************************************"