# Experiment Log

> Retrospective experiment log reconstructed from the current repository snapshot.

## Experiment Block: Inference-Only Baselines

**Date**: 2026-04-10  
**Goal**: Establish non-training baselines for Tangut short-text translation.

### Setup
- **Methods**: `baseline1`, `baseline2`, `baseline2_1_cot`
- **Model**: `qwen2.5-7b-instruct`
- **Test data**: `data/eval/test_set.jsonl` (`50` examples)
- **Metrics**: Lexical coverage, perplexity, chrF++, dictionary-conditioned LLM judge

### Results

| Method | Lex ↑ | PPL ↓ | chrF++ ↑ | LLM Sem ↑ | LLM Flu ↑ | Verdict |
|--------|------:|------:|---------:|----------:|----------:|---------|
| `baseline1` | 0.1165 | 20.21 | 0.19 | 1.00 | 1.56 | Fails as a translation baseline; useful only as a lower bound |
| `baseline2` | 0.8187 | 4474.44 | 6.13 | 2.30 | 2.40 | Strongest inference-only system, but verbose and reference-misaligned |
| `baseline2_1_cot` | 0.8723 | 18119.16 | 7.59 | 2.30 | 1.88 | CoT increases coverage/chrF a bit, but hurts well-formedness |

### Verdict
- **Supports claim?** Partially
- **Key takeaway**: Dictionary prompting is a strong non-training baseline, but it should not be mistaken for the best translation system because it over-optimizes coverage-like metrics.

---

## Experiment Block: Synthetic SFT Variants

**Date**: 2026-04-10  
**Goal**: Compare different synthetic-data construction strategies under the same base model.

### Setup
- **Methods**: `baseline3`, `baseline3_1_unk`, `baseline3_2_multitask`, `baseline3_3_semantic`
- **Real data**: `400` train / `41` dev / `50` test
- **Synthetic data**: `50,000` samples per variant
- **Combined SFT data**: `54,000` samples per variant

### Results

| Method | Lex ↑ | PPL ↓ | chrF++ ↑ | Exact Match ↑ | Notes |
|--------|------:|------:|---------:|--------------:|-------|
| `baseline3` | 0.3192 | 298684.91 | 17.85 | 1/50 | Mixed-input baseline |
| `baseline3_1_unk` | 0.3514 | 11031.18 | 22.11 | 3/50 | Pure-input baseline; strong reference alignment |
| `baseline3_2_multitask` | 0.4839 | 3322.93 | 25.99 | 5/50 | Best current main model after cleaning tags |
| `baseline3_3_semantic` | 0.3723 | 17064.38 | 17.03 | 1/50 | Semantic projection does not beat UNK |

### Verdict
- **Supports claim?** Yes
- **Key takeaway**: Explicit structural alignment in the target text is the strongest current design choice. Pure input helps; noisy semantic projection does not yet pay off.

---

## Experiment Block: Preference Optimization

**Date**: 2026-04-10  
**Goal**: Test whether DPO can improve on the best SFT base selected by the pipeline.

### Setup
- **DPO base selection**: select best SFT among `mixed`, `unk`, and `semantic` by chrF
- **Selected base in current snapshot**: `baseline3_1_unk`
- **Preference pairs**: `3,867`
- **Reward**: lexical coverage minus weighted log perplexity

### Results

| Method | Lex ↑ | PPL ↓ | chrF++ ↑ | Exact Match ↑ | Contamination ↓ |
|--------|------:|------:|---------:|--------------:|----------------:|
| `baseline3_1_unk` | 0.3514 | 11031.18 | 22.11 | 3/50 | 3/50 |
| `final_v2` | 0.4224 | 314577.16 | 19.39 | 2/50 | 9/50 |
| `final` | 0.3913 | 158427.71 | 9.23 | 0/50 | 17/50 |

### Verdict
- **Supports claim?** No
- **Key takeaway**: The current DPO design is unstable and degrades reference alignment.

---

## Experiment Block: Human-Reference Metric Sanity Check

**Date**: 2026-04-10  
**Goal**: Test whether automatic metrics rank the gold reference correctly.

### Setup
- **Method**: feed the reference translation back into the same evaluation pipeline as `human_reference`
- **Purpose**: metric sanity anchor

### Results

| Method | Lex ↑ | PPL ↓ | chrF++ ↑ | LLM Sem ↑ | LLM Flu ↑ |
|--------|------:|------:|---------:|----------:|----------:|
| `human_reference` | 0.5733 | 1646139.34 | 100.00 | 2.24 | 2.34 |

### Verdict
- **Supports claim?** Yes
- **Key takeaway**: chrF++ correctly rewards the gold reference, but lexical coverage and modern-Chinese perplexity do not. These metrics are diagnostic, not authoritative.

---

## Experiment Block: Reference-Aware Judge Rerun

**Date**: 2026-04-10  
**Goal**: Re-evaluate key systems with a judge that sees the Tangut input, the gold reference, and the candidate output simultaneously.

### Setup
- **Methods**: `baseline2`, `baseline3_1_unk`, `baseline3_2_multitask`, `final_v2`, `human_reference`
- **Judge dimensions**:
  - reference agreement,
  - source faithfulness,
  - title-style fitness,
  - overall quality
- **Scale**: `1-5`

### Results

| Method | Ref Agr ↑ | Src Faith ↑ | Title Style ↑ | Overall ↑ | Verdict |
|--------|----------:|------------:|--------------:|----------:|---------|
| `baseline2` | 1.50 | 1.52 | 2.10 | 1.50 | Verbose and reference-misaligned |
| `baseline3_1_unk` | 1.70 | 1.80 | 4.50 | 1.76 | Very title-like, but weaker content recovery |
| `baseline3_2_multitask` | **1.92** | **2.08** | 4.24 | **2.06** | Best non-reference system |
| `final_v2` | 1.64 | 1.78 | 3.72 | 1.72 | Worse than multitask; still unstable |
| `human_reference` | 5.00 | 5.00 | 5.00 | 5.00 | Correctly ranked as perfect |

### Verdict
- **Supports claim?** Yes
- **Key takeaway**: the reference-aware judge validates the main system ranking and fixes the human-reference anomaly.
- **Subset note**: on the `27` suffix-bearing titles, `baseline3_2_multitask` raises exact match from `7.4%` to `14.8%` relative to `baseline3_1_unk`, improves mean overall judge score from `1.93` to `2.30`, and reduces contamination from `11.1%` to `0.0%`.

---

## Experiment Block: DPO Pair Audit

**Date**: 2026-04-10  
**Goal**: Estimate how noisy the preference data is before launching more DPO training.

### Setup
- **Data**: `data/dpo/dpo_pairs.jsonl`
- **Checks**:
  - duplicate chosen/rejected pairs,
  - non-finite rewards,
  - reward-gap distribution,
  - quick similarity proxy to the original synthetic target

### Results

| Check | Result |
|------|--------|
| Total preference pairs | `3,867` |
| Duplicate chosen/rejected pairs | `3` |
| Non-finite reward entries | `3` |
| Mean reward gap | `0.2396` |
| Chosen closer to gold by SequenceMatcher | `71.2%` |
| Chosen closer to gold when reward gap <= 0.1 | `62.3%` |
| Kept pairs if gap >= 0.2 | `1,996` |
| Kept pairs if gap >= 0.4 | `498` |

### Verdict
- **Supports claim?** Partial
- **Key takeaway**: the reward is informative but noisy; a stricter gap threshold is a plausible low-cost follow-up experiment.
