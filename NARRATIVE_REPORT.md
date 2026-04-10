# Narrative Report: Structured Local Adaptation, Frontier Prompting, and Metric Pitfalls in Ultra-Low-Resource Tangut Short-Text Translation

## Core Story

This project studies Tangut-to-Chinese translation in an extremely low-resource setting. The repository contains only `491` real Tangut-Chinese pairs, split into `400` training, `41` development, and `50` test examples. The real examples are short and often title-like rather than full modern sentences, which makes evaluation unusually delicate: exact lexical overlap matters, but title style also breaks common fluency assumptions built into generic Chinese language models.

We now compare four families of methods: local inference-only baselines, a stronger proprietary frontier prompt-only baseline, synthetic-data SFT variants, and DPO-based preference alignment. The strongest reviewer objection was reasonable: maybe a strong SOTA Chinese LLM with carefully engineered CoT-style prompting already solves the task, making the local training story unnecessary. The new DeepSeek-V3.2 few-shot dictionary baseline answers that objection directly. It is indeed the strongest non-reference system on the reference-aware judge, reaching **2.80/5 overall**, **12/50 exact matches**, **24/27** title-suffix preservations, and **0/50** contamination.

That result changes the paper story, but it does not destroy it. The paper should no longer claim that strong prompt-only frontier models fail. The defensible and valuable story is now:

1. **Frontier prompting is a strong proprietary upper bound.**
2. **Within the open/local trainable regime, structured multitask SFT is the best pure SFT base, and strict gap-filtered DPO is the best local learned approach.**
3. **The frontier comparison sharpens the science because the two regimes fail differently.**
4. **Reference-free metrics and noisy preference labels still mislead badly enough to distort conclusions if used naively.**

The best local model, `final_gap04_multitask_sigmoid`, still reaches **chrF++ 30.01**, above the frontier prompt-only baseline's **28.54**, while keeping **5/50 exact matches**, **22/27** suffix preservations, and a near-reference length ratio of **1.12**. It also wins a non-trivial minority of direct pairwise comparisons against the frontier system (`7` wins, `23` ties, `20` losses), especially on short, elliptical, or lexically awkward titles where prompt-only reasoning over-normalizes or hallucinates.

This makes the paper stronger than a local-only benchmark. It becomes a careful comparison of what prompting already solves, what local parameter adaptation still buys under only `491` real pairs, and which metrics can actually be trusted in this narrow historical-title setting.

## Claims

1. **A strong frontier prompt-only baseline is the strongest overall non-reference system, but strict gap-filtered DPO is still the strongest local learned approach and even beats the frontier baseline on chrF++.**
   - Evidence: `frontier_deepseek_v32_fewshot_cot` reaches reference-aware overall `2.80`, exact match `12/50`, and contamination `0/50`.
   - Evidence: `final_gap04_multitask_sigmoid` reaches chrF++ `30.01` versus frontier `28.54`, and wins `7/50` direct pairwise cases under the reference-aware judge.

2. **Structured multitask synthetic SFT is still the strongest pure SFT base.**
   - Evidence: `baseline3_2_multitask` remains the best pure SFT baseline on chrF++ (`25.99`), exact match (`5/50`), and reference-aware overall (`1.98`) among non-DPO local models.

3. **Reference-free metrics mis-rank the gold reference and should not be used as the primary model-selection signal.**
   - Evidence: `human_reference` obtains chrF++ `100.00`, but lexical coverage only `0.5733` and perplexity `1,646,139.34`.

4. **DPO with a lexical-coverage-plus-perplexity reward is useful only after stronger base selection and aggressive pair filtering.**
   - Evidence: legacy `final_v2` remains a negative control, while multitask-base gap-`0.4` runs cleanly outperform both the legacy DPO checkpoint and the multitask SFT baseline on the key combined local metrics.

5. **Frontier prompting and local adaptation fail differently, which is itself a publishable insight.**
   - Evidence: frontier prompting dominates canonical title reconstruction such as `百千印陀羅尼經`, `同音`, `類林`, and `地藏菩薩本願經`.
   - Evidence: the best local model still wins hard short-tail cases such as `五部經`, `正理滴之句義顯具`, `到賢`, and `番言金剛王乘根`.

## Experiments

### Setup

- **Real Tangut data**: `491` pairs total
- **Split**: `400` train / `41` dev / `50` test
- **Synthetic data**: `50,000` samples per SFT variant
- **Combined training size**: `54,000` samples per variant
- **Local base model**: `Qwen2.5-7B-Instruct`
- **Diagnostic LM for perplexity**: `Qwen2.5-0.5B`
- **Frontier model**: `deepseek/deepseek-v3.2` via OpenRouter
- **Method families**:
  - local inference-only: `baseline1`, `baseline2`, `baseline2_1_cot`
  - frontier prompt-only: `frontier_deepseek_v32_fewshot_cot`
  - synthetic SFT: `baseline3`, `baseline3_1_unk`, `baseline3_2_multitask`, `baseline3_3_semantic`
  - preference alignment: `final`, `final_v2`, multitask-base gap-filtered DPO runs

### Experiment 1: Main Comparison on Repository Metrics

| Method | Lex ↑ | PPL ↓ | chrF++ ↑ | LLM Sem ↑ | LLM Flu ↑ |
|--------|------:|------:|---------:|----------:|----------:|
| `baseline1` | 0.1165 | 20.21 | 0.19 | 1.00 | 1.56 |
| `baseline2` | 0.8187 | 4474.44 | 6.13 | 2.30 | 2.40 |
| `baseline2_1_cot` | 0.8723 | 18119.16 | 7.59 | 2.30 | 1.88 |
| `frontier_deepseek_v32_fewshot_cot` | 0.6234 | 7438.72 | 28.54 | 2.62 | 2.78 |
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
- The DeepSeek frontier baseline is the strongest prompt-only system by a large margin and immediately falsifies any paper story that depends on ``stronger CoT still fails.''
- Structured multitask SFT remains the strongest pure SFT baseline.
- Strict gap-`0.4` multitask-base DPO now gives the best local learned raw-overlap result.
- Human-reference scores still show that lexical coverage and PPL are not trustworthy primary metrics for this task.

### Experiment 2: Reference-Aligned Diagnostics from Stored Predictions

| Method | Exact Match ↑ | Contamination ↓ | Length Ratio ↓ | Title-Suffix Preservation ↑ |
|--------|--------------:|----------------:|---------------:|----------------------------:|
| `baseline2` | 0/50 | 0/50 | 1.83 | 3/27 |
| `frontier_deepseek_v32_fewshot_cot` | 12/50 | 0/50 | 0.95 | 24/27 |
| `baseline3_1_unk` | 3/50 | 3/50 | 1.05 | 17/27 |
| `baseline3_2_multitask` | 5/50 | 1/50 | 1.10 | 19/27 |
| `final_v2` | 2/50 | 9/50 | 1.88 | 18/27 |
| `final_gap02_multitask_sigmoid` | 5/50 | 0/50 | 1.74 | 19/27 |
| `final_gap02_multitask_robustwpo` | 4/50 | 7/50 | 1.58 | 23/27 |
| `final_gap04_multitask_sigmoid` | 5/50 | 2/50 | 1.12 | 22/27 |
| `final_gap04_multitask_robustwpo` | 5/50 | 2/50 | 1.26 | 22/27 |
| `human_reference` | 50/50 | 0/50 | 1.00 | 27/27 |

**Interpretation**:
- The frontier baseline is clean, compact, and genuinely title-like. It is not just a verbose dictionary prompt with a stronger model.
- `baseline3_2_multitask` remains the best pure SFT system.
- The gap-`0.2` DPO models prove that DPO can help, but they also expose unstable stopping and contamination.
- The gap-`0.4` DPO models keep most of the gains while restoring near-reference length and low contamination.

### Experiment 3: Reference-Aware Judge Rerun

| Method | Ref Agr ↑ | Src Faith ↑ | Title Style ↑ | Overall ↑ |
|--------|----------:|------------:|--------------:|----------:|
| `baseline2` | 1.50 | 1.52 | 2.10 | 1.50 |
| `frontier_deepseek_v32_fewshot_cot` | 2.78 | 2.82 | 4.48 | **2.80** |
| `baseline3_1_unk` | 1.78 | 1.84 | 4.36 | 1.82 |
| `baseline3_2_multitask` | 1.92 | 1.98 | 4.18 | 1.98 |
| `final_v2` | 1.64 | 1.72 | 3.80 | 1.68 |
| `final_gap02_multitask_sigmoid` | 2.08 | 2.24 | 4.14 | 2.22 |
| `final_gap02_multitask_robustwpo` | 2.10 | 2.16 | 3.78 | 2.12 |
| `final_gap04_multitask_sigmoid` | **2.12** | **2.26** | 4.60 | 2.22 |
| `final_gap04_multitask_robustwpo` | 2.10 | 2.20 | **4.70** | 2.22 |
| `human_reference` | 5.00 | 5.00 | 5.00 | 5.00 |

**Interpretation**:
- The reference-aware judge fixes the human-reference anomaly completely.
- The DeepSeek frontier baseline is the strongest non-reference system overall.
- Within the local learned family, the top overall score is shared by three multitask-base DPO variants.
- The strict gap-`0.4` variants remain the strongest local title reconstructions, while the looser gap-`0.2` sigmoid variant remains strongest on some non-suffix adequacy cases.

### Experiment 4: Human Reference as a Metric Stress Test

The `human_reference` setting feeds the gold translation back into the same evaluation pipeline. A valid primary metric should rank this system at or near the top. chrF++ behaves correctly, but the other metrics do not:

- **Lexical coverage** is only `0.5733`, because dictionary semantics and title wording do not align one-to-one.
- **Perplexity** is extremely poor (`1,646,139.34`), suggesting that the modern-Chinese LM penalizes title style, historical transliteration, and compact book-name formatting.
- **Legacy LLM judge** scores remain only moderate, indicating that the current reference-free judging prompt is not sufficiently discriminative.

This experiment still motivates a key paper decision: **use chrF++ and reference-aware diagnostics as the primary evidence, and move lexical coverage / PPL / legacy judge scores into a diagnostic subsection.**

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

### Experiment 6: Frontier-vs-Local Case Study

Direct pairwise comparison between `frontier_deepseek_v32_fewshot_cot` and `final_gap04_multitask_sigmoid` gives:

- **Overall**: `20` frontier wins / `23` ties / `7` local wins
- **Title-suffix subset**: `12 / 11 / 4`
- **No-title-suffix subset**: `8 / 12 / 3`

The qualitative split is structured rather than random.

**Where the frontier model strongly wins**:
- `百千印陀羅尼經`
- `同音`
- `類林`
- `地藏菩薩本願經`

These are mostly canonical, bibliographically regular titles where broad Chinese title priors help.

**Where the best local model still wins**:
- `五部經`
  - frontier: `五類經`
  - local: `五部經典`
- `正理滴之句義顯具`
  - frontier: `德宜點義句顯宣論`
  - local: `正理滴之句義顯了`
- `到賢`
  - frontier: `五台山觀音普賢殿寺`
  - local: `達賢`
- `番言金剛王乘根`
  - frontier: `番`
  - local: `番言金剛王乘根`

**Interpretation**:
- The frontier model is better overall, so we should say that plainly.
- The local model still occupies a real niche: it remains more robust on some ultra-short, lexically awkward, or non-canonical items where prompt-only reasoning over-regularizes or hallucinates.
- This is the most honest answer to ``why not just use a stronger CoT model?'': use it if you can, but the local training story remains scientifically meaningful because it changes the failure mode and remains competitive on raw overlap.

## Figures

1. **Figure 1**: Method taxonomy figure showing four tracks: local prompting, frontier prompting, synthetic SFT, and DPO.
2. **Table 1**: Main metric comparison including the frontier DeepSeek row.
3. **Table 2**: Reference-aligned diagnostics including the frontier DeepSeek row.
4. **Table 3**: Reference-aware judge comparison including the frontier DeepSeek row.
5. **Figure 2**: Human-reference anchor comparison showing metric misalignment.
6. **Figure 3**: DPO pair-quality audit by reward-gap bins.
7. **Figure 4**: Frontier-vs-local pairwise win/tie/loss chart by subset.

## Known Weaknesses

- The real Tangut dataset is extremely small (`491` examples total).
- The test set is short and title-like, so the paper must avoid overclaiming general sentence translation.
- The paper can no longer claim that the local system is strongest overall against a strong frontier prompt-only baseline.
- The local DPO story is still subset- and metric-sensitive: strict filtering helps canonical title reconstruction most, while looser filtering remains stronger on some non-suffix cases.
- The related-work and citation scaffolding still need a fully verified pass before submission.

## Related Work

- **Ultra-low-resource translation and historical scripts**: already scaffolded in the draft, but still needs final citation verification.
- **Dictionary-augmented inference / lexicon-grounded prompting**: relevant to both the local prompt baselines and the frontier DeepSeek comparison.
- **Synthetic parallel data for low-resource translation**: motivates the SFT family.
- **Preference optimization for language generation and MT**: motivates the DPO family and the noisy-preference discussion.
- **Evaluation for short-form title translation and historical-language outputs**: directly relevant to the human-reference anchor and the reference-aware rerun.

## Proposed Title

Structured Local Adaptation, Frontier Prompting, and Metric Pitfalls in Ultra-Low-Resource Tangut Short-Text Translation

## Target Venue

ACL Findings 2026 (provisional)
