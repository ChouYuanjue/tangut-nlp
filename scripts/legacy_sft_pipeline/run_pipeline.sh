#!/bin/bash
set -e
cd "$(dirname "$0")/../.."
PROJECT_DIR=$(pwd)

echo "=============================================="
echo " Tangut-NLP Full Pipeline"
echo "=============================================="

echo ""
echo "=== PHASE 2: Prepare Data Splits ==="
python3 src/prepare_splits.py \
    --input data/raw/tangut_output.jsonl \
    --test-out data/eval/test_set.jsonl \
    --dev-out data/eval/dev_set.jsonl \
    --train-out data/sft/babelstone_sft.jsonl \
    --test-size 50 --dev-size 41 --seed 42

echo ""
echo "=== PHASE 3: Baseline 1 - Zero-shot ==="
python3 experiments/baseline1_zeroshot.py
python3 eval/run_all_metrics.py \
    --predictions results/baseline1/predictions.jsonl \
    --test-set data/eval/test_set.jsonl \
    --reward-dict data/dictionary/reward_dict.json \
    --output results/baseline1/metrics.json

echo ""
echo "=== PHASE 4: Baseline 2 - Dictionary RAG ==="
python3 experiments/baseline2_dict_rag.py
python3 eval/run_all_metrics.py \
    --predictions results/baseline2/predictions.jsonl \
    --test-set data/eval/test_set.jsonl \
    --reward-dict data/dictionary/reward_dict.json \
    --output results/baseline2/metrics.json

echo ""
echo "=== PHASE 5A: Generate Synthetic Data ==="
python3 src/data_synthesis.py --max-samples 50000

echo ""
echo "=== PHASE 5B: Combine Real + Synthetic Data ==="
python3 src/combine_data.py \
    --real data/sft/babelstone_sft.jsonl \
    --synthetic data/sft/synthetic_sft.jsonl \
    --output data/sft/combined_sft.jsonl \
    --upsample-real 10

echo ""
echo "=== PHASE 5C: SFT Training ==="
accelerate launch --num_processes=2 --mixed_precision=bf16 \
    experiments/baseline3_synthetic_sft.py

echo ""
echo "=== PHASE 5D: Merge LoRA + SFT Inference ==="
python3 experiments/inference.py \
    --model checkpoints/sft/final \
    --test-set data/eval/test_set.jsonl \
    --output results/baseline3/predictions.jsonl \
    --method-name baseline3_sft
python3 eval/run_all_metrics.py \
    --predictions results/baseline3/predictions.jsonl \
    --test-set data/eval/test_set.jsonl \
    --reward-dict data/dictionary/reward_dict.json \
    --output results/baseline3/metrics.json

echo ""
echo "=== PHASE 6A: Generate DPO Candidates ==="
python3 experiments/generate_candidates.py \
    --sft-model checkpoints/sft/merged

echo ""
echo "=== PHASE 6B: DPO Training ==="
accelerate launch --num_processes=2 --mixed_precision=bf16 \
    experiments/final_dpo.py \
    --sft-model checkpoints/sft/merged

echo ""
echo "=== PHASE 6C: DPO Inference ==="
python3 experiments/inference.py \
    --model checkpoints/dpo/final \
    --test-set data/eval/test_set.jsonl \
    --output results/final/predictions.jsonl \
    --method-name final_dpo
python3 eval/run_all_metrics.py \
    --predictions results/final/predictions.jsonl \
    --test-set data/eval/test_set.jsonl \
    --reward-dict data/dictionary/reward_dict.json \
    --output results/final/metrics.json

echo ""
echo "=== PHASE 7: Aggregate Results ==="
python3 eval/aggregate_results.py

echo ""
echo "=============================================="
echo " PIPELINE COMPLETE"
echo "=============================================="
