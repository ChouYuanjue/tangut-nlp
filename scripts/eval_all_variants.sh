#!/bin/bash
set -e
cd "$(dirname "$0")/.."
PROJECT_DIR=$(pwd)

echo "=========================================================="
echo "      Completing Missing Evaluation for All Variants      "
echo "=========================================================="

# 1. Baseline 2.1 (CoT RAG) - Ensure metrics are generated
echo ""
echo "=== EVALUATING: Baseline 2.1 (CoT RAG) ==="
if [ -f "results/baseline2_1_cot/predictions.jsonl" ]; then
    # 2.1 CoT standardly puts the clean result in 'prediction', 
    # but we'll run it through the cleaner anyway for consistency if it had tags
    python3 eval/clean_predictions.py \
        --input results/baseline2_1_cot/predictions.jsonl \
        --output results/baseline2_1_cot/predictions_cleaned.jsonl

    python3 eval/run_all_metrics.py \
        --predictions results/baseline2_1_cot/predictions_cleaned.jsonl \
        --test-set data/eval/test_set.jsonl \
        --reward-dict data/dictionary/reward_dict.json \
        --output results/baseline2_1_cot/metrics.json
else
    echo "Warning: results/baseline2_1_cot/predictions.jsonl not found. Skipping."
fi

# 2. Baseline 3.1 (100% UNK SFT)
echo ""
echo "=== EVALUATING: Baseline 3.1 (100% UNK SFT) ==="
mkdir -p results/baseline3_1_unk
python3 experiments/inference.py \
    --model checkpoints/sft_unk/final \
    --test-set data/eval/test_set.jsonl \
    --output results/baseline3_1_unk/predictions.jsonl \
    --method-name baseline3_1_unk
python3 eval/run_all_metrics.py \
    --predictions results/baseline3_1_unk/predictions.jsonl \
    --test-set data/eval/test_set.jsonl \
    --reward-dict data/dictionary/reward_dict.json \
    --output results/baseline3_1_unk/metrics.json

# 3. Baseline 3.3 (Semantic Projection SFT)
echo ""
echo "=== EVALUATING: Baseline 3.3 (Semantic SFT) ==="
mkdir -p results/baseline3_3_semantic
# Using the latest checkpoint-3000
python3 experiments/inference.py \
    --model checkpoints/sft_semantic/checkpoint-3000 \
    --test-set data/eval/test_set.jsonl \
    --output results/baseline3_3_semantic/predictions.jsonl \
    --method-name baseline3_3_semantic
python3 eval/run_all_metrics.py \
    --predictions results/baseline3_3_semantic/predictions.jsonl \
    --test-set data/eval/test_set.jsonl \
    --reward-dict data/dictionary/reward_dict.json \
    --output results/baseline3_3_semantic/metrics.json

# 4. Final Aggregation
echo ""
echo "=== REGENERATING COMPARISON DASHBOARD ==="
python3 eval/aggregate_results.py

echo ""
echo "=========================================================="
echo " ALL PIPELINES EVALUATED AND AGGREGATED "
echo "=========================================================="
