#!/bin/bash
set -e
cd "$(dirname "$0")/.."

echo "=========================================================="
echo " Starting Baseline 3.2: Multitask Alignment SFT "
echo "=========================================================="

# 1. Generate Multitask Synthetic Data
echo ""
echo "=== STEP 1: Generating Multitask Synthetic Data ==="
python3 src/data_synthesis.py \
    --mode multitask \
    --max-samples 50000 \
    --output data/sft/synthetic_sft_multitask.jsonl

# 2. Combine with Real Data (upsampled)
echo ""
echo "=== STEP 2: Combining Data ==="
python3 src/combine_data.py \
    --real data/sft/babelstone_sft.jsonl \
    --synthetic data/sft/synthetic_sft_multitask.jsonl \
    --output data/sft/combined_sft_multitask.jsonl \
    --upsample-real 10

# 3. SFT Training
echo ""
echo "=== STEP 3: Training Baseline 3.2 (Multitask) ==="
# Using accelerate to handle 2 GPUs
accelerate launch --num_processes=2 --mixed_precision=bf16 \
    experiments/baseline3_synthetic_sft.py \
    --train-data data/sft/combined_sft_multitask.jsonl \
    --output-dir checkpoints/sft_multitask \
    --epochs 3

# 4. Inference
echo ""
echo "=== STEP 4: Inference (Baseline 3.2) ==="
mkdir -p results/baseline3_2_multitask
python3 experiments/inference.py \
    --model checkpoints/sft_multitask/final \
    --test-set data/eval/test_set.jsonl \
    --output results/baseline3_2_multitask/predictions.jsonl \
    --method-name baseline3_2_multitask

# 5. Evaluation (Cleaned for fairness)
echo ""
echo "=== STEP 5: Evaluation (Baseline 3.2, Cleaned) ==="
# We clean the multitask tags (<dict_match>, <literal>) before scoring
python3 eval/clean_predictions.py \
    --input results/baseline3_2_multitask/predictions.jsonl \
    --output results/baseline3_2_multitask/predictions_cleaned.jsonl

python3 eval/run_all_metrics.py \
    --predictions results/baseline3_2_multitask/predictions_cleaned.jsonl \
    --test-set data/eval/test_set.jsonl \
    --reward-dict data/dictionary/reward_dict.json \
    --output results/baseline3_2_multitask/metrics.json

# 6. Final Aggregate
echo ""
echo "=== FINAL: Aggregating All Results ==="
python3 eval/aggregate_results.py

echo ""
echo "=========================================================="
echo " Baseline 3.2 Experiment Complete! "
echo "=========================================================="
