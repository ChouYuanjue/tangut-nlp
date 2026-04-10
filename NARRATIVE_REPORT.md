# Narrative Report: Structured Synthetic Supervision and Metric Pitfalls in Ultra-Low-Resource Tangut Short-Text Translation

## Core Story

This project studies Tangut-to-Chinese translation in an extremely low-resource setting. The repository contains only `491` real Tangut-Chinese pairs, split into `400` training, `41` development, and `50` test examples. The real examples are short and often title-like rather than full modern sentences, which makes evaluation unusually delicate: exact lexical overlap matters, but title style also breaks common fluency assumptions built into generic Chinese language models.

We compare three families of methods: inference-only baselines, synthetic-data SFT variants, and DPO-based preference alignment. The current evidence supports a more specific positive result than before. **Structured multitask SFT** is still the strongest pure supervised baseline, but the strongest overall learned systems appear only after DPO is moved onto that multitask base and the preference pairs are filtered much more aggressively. The strict gap-$0.4$ sigmoid variant reaches **chrF++ 30.01**, keeps **5/50 exact matches**, preserves **22/27** title suffixes, and ties for the best reference-aware overall score (**2.22/5**).

At the same time, the project contains two valuable negative findings. First, the current **reference-free evaluation stack is partially misaligned** with the task: the gold reference itself receives only moderate dictionary coverage and extremely poor perplexity under a modern-Chinese language model, even though chrF++ is perfect. Second, **preference optimization is highly conditional**: legacy `final_v2` remains worse than its SFT base, and the looser gap-$0.2$ multitask DPO variants still expose large length or contamination costs. This makes the repository a strong basis for a paper about both **what works** (structured supervision plus high-quality pair filtering) and **what fails** (misaligned reward-based alignment and mis-specified automatic metrics).

## Claims

1. **Structured multitask synthetic SFT is the strongest pure SFT base, but strict gap-filtered DPO is now the strongest overall learned approach.**
   - Evidence: `baseline3_2_multitask` remains the best pure SFT baseline, while `final_gap04_multitask_sigmoid` reaches chrF++ `30.01` and reference-aware overall `2.22`.

2. **Pure-input training is more helpful than noisy semantic projection in the current setup.**
   - Evidence: `baseline3_1_unk` outperforms `baseline3_3_semantic` on chrF++ (`22.11` vs `17.03`) and exact match (`3/50` vs `1/50`).

3. **Reference-free metrics mis-rank the gold reference and should not be used as the primary model-selection signal.**
   - Evidence: `human_reference` obtains chrF++ `100.00`, but lexical coverage `0.5733` and perplexity `1,646,139.34`.

4. **DPO with a lexical-coverage-plus-perplexity reward is useful only after stronger base selection and aggressive pair filtering.**
   - Evidence: `final_v2` remains a negative control, while the multitask-base gap-$0.4$ runs cleanly outperform both the legacy DPO checkpoint and the multitask SFT baseline on the key combined metrics.

## Experiments

### Setup

- **Real Tangut data**: `491` pairs total
- **Split**: `400` train / `41` dev / `50` test
- **Synthetic data**: `50,000` samples per SFT variant
- **Combined training size**: `54,000` samples per variant
- **Base model**: `Qwen2.5-7B-Instruct`
- **Diagnostic LM for perplexity**: `Qwen2.5-0.5B`
- **Method families**:
  - inference-only: `baseline1`, `baseline2`, `baseline2_1_cot`
  - synthetic SFT: `baseline3`, `baseline3_1_unk`, `baseline3_2_multitask`, `baseline3_3_semantic`
  - preference alignment: `final`, `final_v2`

### Experiment 1: Main Comparison on Repository Metrics

| Method | Lex ↑ | PPL ↓ | chrF++ ↑ | LLM Sem ↑ | LLM Flu ↑ |
|--------|------:|------:|---------:|----------:|----------:|
| `baseline1` | 0.1165 | 20.21 | 0.19 | 1.00 | 1.56 |
| `baseline2` | 0.8187 | 4474.44 | 6.13 | 2.30 | 2.40 |
| `baseline2_1_cot` | 0.8723 | 18119.16 | 7.59 | 2.30 | 1.88 |
| `baseline3` | 0.3192 | 298684.91 | 17.85 | 1.42 | 1.88 |
| `baseline3_1_unk` | 0.3514 | 11031.18 | 22.11 | 1.66 | 1.96 |
| `baseline3_2_multitask` | 0.4839 | 3322.93 | 25.99 | 1.70 | 2.26 |
| `baseline3_3_semantic` | 0.3723 | 17064.38 | 17.03 | 1.46 | 1.84 |
| `final` | 0.3913 | 158427.71 | 9.23 | 1.40 | 1.42 |
| `final_v2` | 0.4224 | 314577.16 | 19.39 | 1.52 | 1.76 |
| `final_gap02_multitask_sigmoid` | 0.5694 | 303438.00 | 20.93 | 1.88 | 1.86 |
| `final_gap02_multitask_robustwpo` | 0.5249 | 238892.42 | 27.67 | 1.88 | 1.88 |
| `final_gap04_multitask_sigmoid` | 0.4833 | 9573.71 | **30.01** | 1.84 | 2.04 |
| `final_gap04_multitask_robustwpo` | 0.4547 | 20012.69 | 27.52 | 1.74 | 2.08 |
| `human_reference` | 0.5733 | 1646139.34 | 100.00 | 2.24 | 2.34 |

**Interpretation**:
- Inference-only prompting is strong on lexical coverage but weak on reference matching.
- Structured multitask SFT remains the strongest pure SFT baseline.
- Strict gap-filtered multitask-base DPO now gives the best overall raw/clean trade-off, while looser filtering remains visibly unstable.
- Human-reference scores reveal that lexical coverage and PPL are not trustworthy primary metrics for this task.

### Experiment 2: Reference-Aligned Diagnostics from Stored Predictions

| Method | Exact Match ↑ | Contamination ↓ | Length Ratio ↓ | Title-Suffix Preservation ↑ |
|--------|--------------:|----------------:|---------------:|----------------------------:|
| `baseline2` | 0/50 | 0/50 | 1.83 | 3/27 |
| `baseline3_1_unk` | 3/50 | 3/50 | 1.05 | 17/27 |
| `baseline3_2_multitask` | 5/50 | 1/50 | 1.10 | 19/27 |
| `final_v2` | 2/50 | 9/50 | 1.88 | 18/27 |
| `final_gap02_multitask_sigmoid` | 5/50 | 0/50 | 1.74 | 19/27 |
| `final_gap02_multitask_robustwpo` | 4/50 | 7/50 | 1.58 | 23/27 |
| `final_gap04_multitask_sigmoid` | 5/50 | 2/50 | 1.12 | 22/27 |
| `final_gap04_multitask_robustwpo` | 5/50 | 2/50 | 1.26 | 22/27 |
| `human_reference` | 50/50 | 0/50 | 1.00 | 27/27 |

**Interpretation**:
- `baseline3_2_multitask` remains the best pure SFT system.
- The gap-0.2 DPO models prove that DPO can help, but they also expose unstable stopping and contamination.
- The gap-0.4 DPO models keep the gains while restoring near-reference length and low contamination.
- Dictionary prompting is often too verbose for title reconstruction, which explains why its lexical coverage is high but chrF++ is low.

### Experiment 3: Reference-Aware Judge Rerun

| Method | Ref Agr ↑ | Src Faith ↑ | Title Style ↑ | Overall ↑ |
|--------|----------:|------------:|--------------:|----------:|
| `baseline2` | 1.50 | 1.52 | 2.10 | 1.50 |
| `baseline3_1_unk` | 1.78 | 1.84 | 4.36 | 1.82 |
| `baseline3_2_multitask` | 1.92 | 1.98 | 4.18 | 1.98 |
| `final_v2` | 1.64 | 1.72 | 3.80 | 1.68 |
| `final_gap02_multitask_sigmoid` | 2.08 | 2.24 | 4.14 | 2.22 |
| `final_gap02_multitask_robustwpo` | 2.10 | 2.16 | 3.78 | 2.12 |
| `final_gap04_multitask_sigmoid` | **2.12** | **2.26** | 4.60 | **2.22** |
| `final_gap04_multitask_robustwpo` | 2.10 | 2.20 | **4.70** | **2.22** |
| `human_reference` | 5.00 | 5.00 | 5.00 | 5.00 |

**Interpretation**:
- The reference-aware judge fixes the human-reference anomaly completely.
- The top overall score is now shared by three multitask-base DPO variants.
- The strict gap-0.4 variants dominate the suffix-bearing titles, while the looser gap-0.2 sigmoid variant remains strongest on non-suffix items.
- The older `final_v2` result remains a negative control, which strengthens the paper's causal story about base-model quality and pair filtering.

### Experiment 4: Human Reference as a Metric Stress Test

The `human_reference` setting feeds the gold translation back into the same evaluation pipeline. A valid primary metric should rank this system at or near the top. chrF++ behaves correctly, but the other metrics do not:

- **Lexical coverage** is only `0.5733`, because dictionary semantics and title wording do not align one-to-one.
- **Perplexity** is extremely poor (`1,646,139.34`), suggesting that the modern-Chinese LM is penalizing title style, historical transliteration, and compact book-name formatting.
- **LLM judge** scores remain only slightly above strong prompting baselines, indicating that the current reference-free judging prompt is not sufficiently discriminative.

This experiment motivates a key paper decision: **use chrF++ and reference-aware diagnostics as the primary evidence, and move lexical coverage / PPL / current judge scores into a diagnostic subsection.**

### Experiment 5: DPO Pair Quality Audit

The DPO pipeline currently constructs `3,867` preference pairs. A quick audit reveals multiple weaknesses:

- `3` pairs contain duplicate chosen/rejected texts.
- `3` entries have non-finite reward values.
- Using a simple similarity proxy against the original synthetic targets, the reward-chosen output is closer to gold only **71.3%** of the time overall.
- The proxy quality rises monotonically with reward gap: **62.3%** for `(0.05, 0.10]`, **66.3%** for `(0.10, 0.20]`, **75.4%** for `(0.20, 0.40]`, and **81.5%** for `>= 0.40`.

**Interpretation**:
- The reward is directionally useful, but noisy.
- A stricter reward-gap threshold is the most plausible low-cost DPO follow-up because it is directly isolating cleaner preference labels.
- The follow-up confirms that the gap threshold is not cosmetic: it is the main stability control for DPO in this repository.

### Experiment 6: Case-Study Summary

The best qualitative behavior now comes from the strict gap-filtered multitask-base DPO runs, especially `final_gap04_multitask_sigmoid`, which reaches the same `5/50` exact matches as the multitask SFT baseline while also improving raw chrF++ and title-suffix preservation.

Representative exact matches still include:

- `集頌般若波羅蜜多經`
- `最上意經`
- `番言金剛王乘根`
- `金剛般若略記文`
- `佛説父母恩重經`

In contrast:

- `baseline2` often over-expands titles into explanatory paraphrases.
- `final_v2` shows mixed-script contamination and unstable generation.
- `final_gap02_multitask_sigmoid` often captures content but overgenerates on short titles.
- `final_gap04_multitask_robustwpo` is more title-like than the looser robust variant, but still slightly less length-stable than the strict sigmoid run.
- `baseline3_3_semantic` does not show reliable gains over the simpler `UNK` approach.

## Figures

1. **Figure 1**: Method taxonomy figure showing the three tracks: prompting, synthetic SFT, and DPO.
2. **Table 1**: Main metric comparison (existing `results/comparison.csv`).
3. **Table 2**: Reference-aligned diagnostics derived from stored predictions.
4. **Figure 2**: Human-reference anchor comparison showing metric misalignment.
5. **Figure 3**: DPO pair-quality audit by reward-gap bins.

## Known Weaknesses

- The real Tangut dataset is extremely small (`491` examples total).
- The test set is short and title-like, so the paper must avoid overclaiming general sentence translation.
- The related-work and citation scaffolding still need verified bibliographic sources.
- The DPO story is now materially stronger, but it is still subset-sensitive: strict filtering helps canonical title reconstruction most, while looser filtering remains stronger on some non-suffix cases.

## Related Work

- **Ultra-low-resource translation and historical scripts**: needs verified citations.
- **Dictionary-augmented inference / lexicon-grounded prompting**: needs verified citations.
- **Synthetic parallel data for low-resource translation**: needs verified citations.
- **Preference optimization for language generation**: needs verified citations.
- **Evaluation for short-form title translation and historical-language outputs**: needs verified citations.

## Proposed Title

Structured Synthetic Supervision and Metric Pitfalls in Ultra-Low-Resource Tangut Short-Text Translation

## Target Venue

ACL Findings 2026 (provisional)
