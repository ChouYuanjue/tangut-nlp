# Supplementary Experiment Plan

> Follow-up plan for turning the current snapshot into a reviewer-ready paper.

## Objective

Convert the current repository snapshot into a paper with:

1. one defensible main positive result,
2. one rigorous evaluation contribution,
3. one well-supported negative result section.

## Main Paper Direction

- **Headline result**: structured multitask synthetic SFT (`baseline3_2_multitask`) is the best current reference-aligned method.
- **Evaluation contribution**: lexical coverage and modern-Chinese perplexity mis-rank the human reference on this title-like Tangut task.
- **Negative result**: DPO with lexical-coverage-plus-perplexity reward is unstable and degrades output quality.

## Milestone 0: No-GPU Analyses (must run first)

### M0.1 Reference-aligned diagnostics from stored predictions
- **Goal**: supplement the existing metric table with paper-friendly diagnostics.
- **Inputs**: existing `results/*/predictions.jsonl`
- **Metrics to derive**:
  - exact match,
  - output contamination rate,
  - average output length / reference length ratio,
  - title-suffix preservation (`經/論/記/疏/頌/品/...`).
- **Success criterion**: produce one clean comparison table to support the main results section.

### M0.2 Human-reference anchor analysis
- **Goal**: formalize why lexical coverage and PPL are only diagnostic.
- **Inputs**: `results/human_reference/metrics.json`
- **Outputs**:
  - a short sanity-check subsection,
  - a figure/table comparing `human_reference` against the main systems.
- **Success criterion**: demonstrate that the gold reference is not top-ranked by lexical coverage or PPL.

### M0.3 DPO pair-quality audit
- **Goal**: quantify how noisy the current preference signal is.
- **Inputs**: `data/dpo/dpo_pairs.jsonl`, original synthetic targets
- **Checks**:
  - invalid pairs,
  - duplicate pairs,
  - reward-gap histogram,
  - gold-similarity proxy by reward-gap bin.
- **Success criterion**: decide whether gap filtering is worth a rerun.

## Milestone 1: Evaluation Repair (low cost)

### M1.1 Reference-aware judge
- **Goal**: replace or supplement the current reference-free LLM judge.
- **Design**:
  - compare `baseline2`, `baseline3_1_unk`, `baseline3_2_multitask`, `final_v2`, and `human_reference`;
  - evaluate with access to the source and the gold reference;
  - ask for faithfulness and title correctness, not free-form fluency.
- **Success criterion**: obtain a ranking that agrees better with chrF++ and human intuition.

### M1.2 Subset analysis
- **Goal**: determine whether methods behave differently on title-like examples.
- **Splits**:
  - examples containing title suffixes,
  - examples without title suffixes,
  - short (`<= 6` chars) vs longer references.
- **Success criterion**: sharpen the task definition in the paper.

## Milestone 2: DPO Follow-ups (GPU, only if M0 supports it)

### M2.1 Gap-filtered DPO on the UNK base
- **Rationale**: low reward-gap pairs are noisier.
- **Variants**:
  - filter with `gap >= 0.2`,
  - filter with `gap >= 0.4`.
- **Budget**: one short pilot each before a full run.
- **Success criterion**: beat `baseline3_1_unk` on chrF++ without increasing contamination.

### M2.2 DPO from the multitask base
- **Rationale**: the current selector never tests DPO from the strongest SFT model.
- **Required changes**:
  - allow `multitask` in base selection,
  - clean structured tags before reward scoring if necessary.
- **Success criterion**: determine whether DPO failure is due to the reward or to the base-model choice.

## Milestone 3: Optional Semantic-Projection Cleanup

### M3.1 Revisit semantic projection only if reviewers ask
- **Rationale**: the current semantic branch is clearly below the UNK baseline.
- **Possible fixes**:
  - restrict projection candidates by part-of-speech-like heuristics,
  - use top-1 instead of vote,
  - add confidence threshold before projection.
- **Default decision**: defer unless needed.

## Compute Budget

- **No-GPU analyses**: immediate
- **Reference-aware judge**: low cost, mostly API
- **Gap-filtered DPO pilots**: moderate
- **Multitask DPO pilot**: moderate to high

## Exit Criteria

The paper is ready for a first full draft when:

- `baseline3_2_multitask` remains the best reference-aligned model,
- the human-reference metric sanity check is documented,
- the DPO section is either repaired by a low-cost rerun or clearly framed as a negative result.
