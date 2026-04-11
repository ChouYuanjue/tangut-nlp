# Paper Plan

## Metadata

- **Title**: Structured Synthetic Supervision and Metric Pitfalls in Ultra-Low-Resource Tangut Short-Text Translation
- **Venue**: ACL Findings 2026 (provisional)
- **One-sentence contribution**: We show that structured synthetic supervision is a stronger path than preference optimization for Tangut short-text translation, and that several reference-free metrics mis-rank the human reference on this task.

## Claims-Evidence Matrix

| # | Claim | Evidence | Section |
|---|-------|----------|---------|
| C1 | Structured multitask synthetic SFT is the strongest current method | `baseline3_2_multitask` has the best non-reference chrF++ and exact match | §4, §5 |
| C2 | Pure-input training is stronger than noisy semantic projection in the current setup | `baseline3_1_unk` > `baseline3_3_semantic` on chrF++ and exact match | §4.3 |
| C3 | Reference-free metrics mis-rank the gold reference | `human_reference` gets chrF++ 100 but weak lexical coverage and extreme PPL | §5.2 |
| C4 | Current DPO is unstable because the preference signal is noisy and misaligned | `final_v2` underperforms its SFT base; DPO pair audit reveals noisy low-gap pairs | §5.3 |

## Section Plan

### 1. Introduction (~1.5 pages)
- **What**: Tangut short-text translation under only 491 real parallel pairs.
- **Why**: Historical-language translation is data-poor, and short title-like strings make evaluation unusually brittle.
- **How**: Compare prompting, synthetic SFT, and DPO under a unified experiment matrix.
- **Result**: Structured multitask synthetic SFT is best; several reference-free metrics fail to rank the human reference correctly.

### 2. Task and Experimental Setting (~1 page)
- Explain the dataset size and split.
- Clarify that the task is short-text / title translation, not open-domain sentence MT.
- Define primary vs diagnostic metrics.

### 3. Methods (~1.5 pages)
- **Prompting track**: zero-shot, dictionary RAG, CoT RAG.
- **Synthetic SFT track**: mixed, UNK, semantic projection, multitask alignment.
- **Preference-alignment track**: DPO from selected SFT base.
- Include a diagram showing the three tracks.

### 4. Main Results (~2 pages)
- Main comparison table.
- Highlight that `baseline3_2_multitask` is the best reference-aligned system.
- Report that `UNK` purity beats semantic projection.
- Position prompting baselines as strong but reference-misaligned.

### 5. Analysis (~2 pages)
- **5.1 Reference-aligned diagnostics**: exact match, contamination, length ratio, suffix preservation.
- **5.2 Metric sanity check**: human-reference anchor.
- **5.3 Failure analysis for DPO**: noisy preference pairs, reward-gap analysis, exclusion of multitask from the current selector.

### 6. Discussion and Limitations (~0.75 pages)
- Narrow task scope.
- Why title-style outputs break generic PPL assumptions.
- What is still missing for stronger DPO claims.

### 7. Conclusion (~0.25 pages)
- Restate the main positive result, the evaluation contribution, and the DPO negative result.

## Figure Plan

| # | Type | Description | Auto? |
|---|------|-------------|:-----:|
| Fig 1 | Method taxonomy | Three-track overview: prompting, synthetic SFT, DPO | manual |
| Table 1 | Main results | Lex, PPL, chrF++, judge scores across all methods | existing results |
| Table 2 | Diagnostic table | Exact match, contamination, length ratio, suffix preservation | scripted |
| Fig 2 | Anchor plot | Human-reference vs model metrics to show misalignment | scripted |
| Fig 3 | Gap plot | Reward-gap bin vs gold-similarity proxy in DPO pairs | scripted |
| Table 3 | DPO audit | Duplicate pairs, invalid rewards, gap-threshold counts | scripted |

## Key References

1. `[VERIFY]` low-resource / historical-language translation survey
2. `[VERIFY]` lexicon-grounded or dictionary-augmented translation
3. `[VERIFY]` synthetic data for low-resource MT
4. `[VERIFY]` DPO / preference optimization
5. `[VERIFY]` evaluation challenges for short-form or historical translation
