# Follow-up Runbook

This runbook turns the current paper draft into a stronger submission with the smallest useful set of additional experiments.

## 1. Recompute paper diagnostics

```bash
python3 scripts/analyze_snapshot.py
```

Outputs:
- `results/paper_diagnostics_reference.json`
- `results/paper_diagnostics_reference.csv`
- `results/paper_diagnostics_dpo.json`

## 2. Run a reference-aware judge

Recommended systems:
- `baseline2`
- `baseline3_1_unk`
- `baseline3_2_multitask`
- `final_v2`
- `human_reference`

Example:

```bash
python3 eval/reference_aware_judge.py \
  --predictions results/baseline3_2_multitask/predictions_cleaned.jsonl \
  --test-set data/eval/test_set.jsonl \
  --output results/baseline3_2_multitask/reference_aware_judge.json
```

For a smoke test without API calls:

```bash
python3 eval/reference_aware_judge.py \
  --predictions results/baseline3_2_multitask/predictions_cleaned.jsonl \
  --test-set data/eval/test_set.jsonl \
  --output results/baseline3_2_multitask/reference_aware_judge_mock.json \
  --mock
```

## 3. Prepare filtered DPO pairs

```bash
python3 scripts/filter_dpo_pairs.py \
  --input data/dpo/dpo_pairs.jsonl \
  --output data/dpo/dpo_pairs_gap02.jsonl \
  --min-gap 0.2 \
  --dedupe
```

Stricter setting:

```bash
python3 scripts/filter_dpo_pairs.py \
  --input data/dpo/dpo_pairs.jsonl \
  --output data/dpo/dpo_pairs_gap04.jsonl \
  --min-gap 0.4 \
  --dedupe
```

Dry-run summary only:

```bash
python3 scripts/filter_dpo_pairs.py \
  --input data/dpo/dpo_pairs.jsonl \
  --output /tmp/unused.jsonl \
  --min-gap 0.2 \
  --dedupe \
  --dry-run
```

## 4. Gap-filtered DPO on the UNK base

```bash
CUDA_VISIBLE_DEVICES=0 python3 experiments/final_dpo.py \
  --dpo-data data/dpo/dpo_pairs_gap02.jsonl \
  --sft-model checkpoints/sft_unk/merged \
  --output-dir checkpoints/dpo_gap02_unk \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 5e-5 \
  --beta 0.1
```

Inference:

```bash
python3 experiments/inference.py \
  --model checkpoints/dpo_gap02_unk/final \
  --test-set data/eval/test_set.jsonl \
  --output results/final_gap02_unk/predictions.jsonl \
  --method-name final_gap02_unk \
  --tensor-parallel 2
```

Evaluation:

```bash
python3 -m eval.run_all_metrics \
  --predictions results/final_gap02_unk/predictions.jsonl \
  --test-set data/eval/test_set.jsonl \
  --reward-dict data/dictionary/reward_dict.json \
  --ppl-model models/qwen2.5-0.5b \
  --output results/final_gap02_unk/metrics.json
```

## 5. DPO from the multitask base

This is the most reviewer-relevant control, because the current selector excludes `multitask`.

```bash
CUDA_VISIBLE_DEVICES=0 python3 experiments/final_dpo.py \
  --dpo-data data/dpo/dpo_pairs_gap02.jsonl \
  --sft-model checkpoints/sft_multitask/merged \
  --output-dir checkpoints/dpo_gap02_multitask \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 8 \
  --lr 5e-5 \
  --beta 0.1
```

Then run inference and evaluation in the same pattern as above.

## 6. Update the paper

After any new result:

1. Append the outcome to `EXPERIMENT_LOG.md`.
2. Update `findings.md`.
3. Refresh `NARRATIVE_REPORT.md`.
4. Sync the LaTeX draft in `paper/`.
