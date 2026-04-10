#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash scripts/wait_and_eval_method.sh \
    --gpu 0 \
    --model checkpoints/dpo_xxx/final \
    --method final_xxx \
    [--poll-seconds 60]
EOF
}

GPU=""
MODEL=""
METHOD=""
POLL_SECONDS="60"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --method)
      METHOD="$2"
      shift 2
      ;;
    --poll-seconds)
      POLL_SECONDS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$GPU" || -z "$MODEL" || -z "$METHOD" ]]; then
  usage >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ "$MODEL" = /* ]]; then
  MODEL_PATH="$MODEL"
else
  MODEL_PATH="$ROOT_DIR/$MODEL"
fi

RESULT_DIR="$ROOT_DIR/results/$METHOD"
PRED_PATH="$RESULT_DIR/predictions.jsonl"
METRICS_PATH="$RESULT_DIR/metrics.json"
JUDGE_PATH="$RESULT_DIR/reference_aware_judge.json"

mkdir -p "$RESULT_DIR"

echo "[$(date -u +%FT%TZ)] Waiting for model path: $MODEL_PATH"
until [[ -f "$MODEL_PATH/adapter_config.json" || -f "$MODEL_PATH/config.json" ]]; do
  sleep "$POLL_SECONDS"
done
echo "[$(date -u +%FT%TZ)] Model ready: $MODEL_PATH"

if [[ ! -f "$PRED_PATH" ]]; then
  echo "[$(date -u +%FT%TZ)] Running inference for $METHOD"
  CUDA_VISIBLE_DEVICES="$GPU" python experiments/inference.py \
    --model "$MODEL_PATH" \
    --test-set data/eval/test_set.jsonl \
    --output "$PRED_PATH" \
    --method-name "$METHOD" \
    --tensor-parallel 1
else
  echo "[$(date -u +%FT%TZ)] Predictions already exist: $PRED_PATH"
fi

if [[ ! -f "$METRICS_PATH" ]]; then
  echo "[$(date -u +%FT%TZ)] Running metrics for $METHOD"
  CUDA_VISIBLE_DEVICES="$GPU" python -m eval.run_all_metrics \
    --predictions "$PRED_PATH" \
    --test-set data/eval/test_set.jsonl \
    --reward-dict data/dictionary/reward_dict.json \
    --ppl-model models/qwen2.5-0.5b \
    --output "$METRICS_PATH"
else
  echo "[$(date -u +%FT%TZ)] Metrics already exist: $METRICS_PATH"
fi

if [[ ! -f "$JUDGE_PATH" ]]; then
  echo "[$(date -u +%FT%TZ)] Running reference-aware judge for $METHOD"
  python eval/reference_aware_judge.py \
    --predictions "$PRED_PATH" \
    --test-set data/eval/test_set.jsonl \
    --output "$JUDGE_PATH"
else
  echo "[$(date -u +%FT%TZ)] Reference-aware judge already exists: $JUDGE_PATH"
fi

echo "[$(date -u +%FT%TZ)] Evaluation bundle complete for $METHOD"
