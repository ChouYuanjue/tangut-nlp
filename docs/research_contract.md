# Research Contract: Structured Synthetic Supervision for Ultra-Low-Resource Tangut Short-Text Translation

> Working contract for the current paper direction. This document narrows the scope of the project to the claims that are currently supported by the repository snapshot.

## Selected Idea

- **Description**: Study Tangut-to-Chinese translation under an extremely low-resource setting using only 491 real Tangut-Chinese pairs, plus synthetic supervision built from dictionary knowledge and monolingual classical Chinese. Compare inference-only prompting, synthetic SFT variants, and DPO-based preference alignment.
- **Source**: Retrospective synthesis of the current repository snapshot (`results/`, `checkpoints/`, `run.sh`, and `eval/`).
- **Selection rationale**: The strongest current evidence supports a focused paper on short, title-like Tangut translation rather than general open-ended machine translation. The current snapshot already contains one positive result (`baseline3_2_multitask`), one informative negative result (DPO), and one important evaluation caveat (`human_reference` is mis-ranked by some automatic metrics).

## Core Claims

1. **Structured multitask synthetic SFT is the strongest reference-aligned method in the current study.**
2. **Pure-input SFT is more helpful than noisy semantic projection in this setting; `UNK`-based purity is currently stronger than vector-space semantic projection.**
3. **Reference-free metrics such as dictionary coverage and modern-Chinese perplexity should be treated as diagnostic signals, not primary selection criteria, because they systematically mis-rank the human reference.**
4. **DPO with a lexical-coverage-plus-perplexity reward is unstable in this setting and should be presented as a negative result / failure analysis rather than the headline contribution.**

## Method Summary

We compare three families of approaches:

1. **Inference-only methods**:
   - `baseline1`: zero-shot translation with the base instruction model.
   - `baseline2`: dictionary-augmented prompting.
   - `baseline2_1_cot`: dictionary-augmented prompting with a three-step CoT-style prompt.

2. **Synthetic-SFT methods**:
   - `baseline3`: mixed Tangut-Chinese synthetic input.
   - `baseline3_1_unk`: 100% pure Tangut input with `UNK` placeholders on dictionary misses.
   - `baseline3_3_semantic`: 100% pure Tangut input with vector-space semantic projection on dictionary misses.
   - `baseline3_2_multitask`: structured SFT with explicit `<dict_match>` and `<literal>` alignment targets.

3. **Preference optimization**:
   - `final`: legacy DPO path.
   - `final_v2`: DPO trained from the best SFT base among `mixed`, `unk`, and `semantic`, selected by chrF. In the current snapshot, the selected base is `UNK`, not `multitask`.

## Experiment Design

- **Real Tangut data**: 491 total pairs from `data/raw/tangut_output.jsonl`
- **Splits**: 400 train / 41 dev / 50 test
- **Synthetic data per variant**: 50,000 pairs
- **Combined SFT data per variant**: 54,000 pairs
- **Base model**: `models/qwen2.5-7b-instruct`
- **Diagnostic language model**: `models/qwen2.5-0.5b`
- **Primary evaluation for paper claims**:
  - chrF++
  - Exact match (derived from stored predictions)
  - Output contamination rate (derived from stored predictions)
  - Title-suffix preservation (derived from stored predictions)
- **Diagnostic-only metrics**:
  - Lexical Coverage
  - Modern-Chinese perplexity
  - Dictionary-conditioned LLM judge

## Baselines

| Method | Type | Main role in paper |
|--------|------|--------------------|
| `baseline1` | Zero-shot | Lower bound / sanity check |
| `baseline2` | Dictionary RAG | Strong inference-only baseline |
| `baseline2_1_cot` | RAG + CoT | Prompt engineering ablation |
| `baseline3` | Mixed synthetic SFT | Original synthetic baseline |
| `baseline3_1_unk` | Pure-input SFT | Strong purity baseline |
| `baseline3_2_multitask` | Structured SFT | Main positive result |
| `baseline3_3_semantic` | Semantic projection SFT | Test whether noisy semantic preservation helps |
| `final_v2` | DPO on selected SFT base | Negative result / failure analysis |
| `human_reference` | Gold output through same evaluation pipeline | Metric sanity anchor |

## Current Results

| Method | Lex ↑ | PPL ↓ | chrF++ ↑ | LLM Sem ↑ | LLM Flu ↑ | Exact Match ↑ |
|--------|------:|------:|---------:|----------:|----------:|--------------:|
| `baseline1` | 0.1165 | 20.21 | 0.19 | 1.00 | 1.56 | 0/50 |
| `baseline2` | 0.8187 | 4474.44 | 6.13 | 2.30 | 2.40 | 0/50 |
| `baseline2_1_cot` | 0.8723 | 18119.16 | 7.59 | 2.30 | 1.88 | 1/50 |
| `baseline3` | 0.3192 | 298684.91 | 17.85 | 1.42 | 1.88 | 1/50 |
| `baseline3_1_unk` | 0.3514 | 11031.18 | 22.11 | 1.66 | 1.96 | 3/50 |
| `baseline3_2_multitask` | 0.4839 | 3322.93 | 25.99 | 1.70 | 2.26 | 5/50 |
| `baseline3_3_semantic` | 0.3723 | 17064.38 | 17.03 | 1.46 | 1.84 | 1/50 |
| `final_v2` | 0.4224 | 314577.16 | 19.39 | 1.52 | 1.76 | 2/50 |
| `human_reference` | 0.5733 | 1646139.34 | 100.00 | 2.24 | 2.34 | 50/50 |

## Key Decisions

- **Scope the task honestly**: this repository currently supports a paper on **Tangut short-text / title translation**, not unconstrained general MT.
- **Make `baseline3_2_multitask` the headline model**: it is the strongest reference-aligned result now available.
- **Move metric criticism into the core narrative**: the `human_reference` anomaly is not a side note; it is a central evaluation finding.
- **Treat DPO as a controlled negative result**: the current `final_v2` setup excludes `multitask` from base selection and uses a reward that is partially misaligned with the task.
- **Prioritize low-cost supplementary analyses before new GPU runs**.

## Status

- [x] Real-data split identified
- [x] Existing baselines audited
- [x] Main claim boundary defined
- [x] Primary positive result identified
- [x] Metric-risk section identified
- [x] Negative-result section identified
- [x] Markdown draft
- [x] Plain LaTeX draft
- [x] Supplementary evaluation rerun
- [ ] DPO follow-up experiment
- [ ] Literature-backed related-work section
- [ ] Venue-specific template migration
