# Findings

> Cross-stage discovery log for the Tangut-NLP paper direction.

---

# Research Findings

## [2026-04-10] Structured multitask SFT is the strongest pure SFT baseline in the current snapshot
- `baseline3_2_multitask` remains the best non-DPO training result and the strongest pure SFT baseline.
- Evidence:
  - chrF++ `25.99`, the highest among pure SFT systems.
  - Exact match `5/50`, higher than `baseline3_1_unk` (`3/50`) and `final_v2` (`2/50`).
  - Output contamination `1/50` after cleaning, much better than legacy or loose-filter DPO systems.
  - Title-suffix preservation `19/27`, which remains strong even after the later DPO follow-ups.
- Implication: the paper should still headline explicit structural alignment as the key base recipe, but no longer claim that DPO is uniformly worse.

## [2026-04-10] Input purity helps more than noisy semantic projection in the current setup
- `baseline3_1_unk` outperforms `baseline3_3_semantic` on chrF++ (`22.11` vs `17.03`) and exact match (`3/50` vs `1/50`).
- Evidence suggests that the current vector-space semantic projection injects more noise than benefit.
- Implication: the semantic-projection branch is useful as a negative ablation, not as the main method.

## [2026-04-10] Dictionary coverage and modern-Chinese perplexity mis-rank the human reference
- `human_reference` obtains chrF++ `100.00`, but lexical coverage only `0.5733` and perplexity `1,646,139.34`.
- `baseline2` and `baseline2_1_cot` score higher on dictionary coverage and LLM-judge semantic scores than several more reference-aligned systems, despite much lower chrF++ and exact match.
- Implication: lexical coverage, PPL, and the current dictionary-conditioned LLM judge must be downgraded to diagnostic metrics.

## [2026-04-10] A reference-aware judge validates the pre-DPO baseline ranking and sharpens the task boundary
- `human_reference` receives perfect `5.0/5.0/5.0/5.0` scores on reference agreement, source faithfulness, title-style fitness, and overall quality.
- Before the multitask-base DPO follow-ups were added, `baseline3_2_multitask` was the strongest non-reference system on reference agreement (`1.92`), source faithfulness (`2.08`), and overall quality (`2.06`).
- `baseline3_1_unk` is slightly more conservative on title shape (`4.50` vs `4.24` title-style fitness), but this does not translate into stronger content recovery.
- On the `27` title-suffix examples, `baseline3_2_multitask` reaches `14.8%` exact match with `0.0%` contamination, versus `7.4%` and `11.1%` for `baseline3_1_unk`.
- Implication: the paper can now argue more strongly that the repository supports Tangut short-title translation specifically, not general sentence translation.

## [2026-04-10] Gap-filtered multitask-base DPO beats the strongest SFT baseline under the fixed-key reference-aware judge
- `final_gap02_multitask_sigmoid` reaches reference-aware overall `2.22`, above `baseline3_2_multitask` (`1.98`) and far above legacy `final_v2` (`1.68`).
- `final_gap02_multitask_robustwpo` also beats the multitask SFT baseline on overall quality (`2.12`) and slightly leads it on reference agreement (`2.10` vs `1.92`).
- The wins are structured rather than uniform:
  - `sigmoid` is strongest overall and strongest on source faithfulness (`2.24`).
  - `robustwpo` is strongest on raw chrF++ (`27.67`) and title-suffix preservation (`23/27`), but its title-style score drops to `3.78` because of contamination.
- Implication: the paper should no longer frame DPO as a blanket negative result. The defensible claim is that DPO helps once the base model and pair quality are controlled, but different objectives expose different failure modes.

## [2026-04-10] Stricter gap-$0.4$ filtering stabilizes multitask-base DPO and produces the strongest overall learned systems
- `final_gap04_multitask_sigmoid` reaches chrF++ `30.01`, reference-aware overall `2.22`, exact match `5/50`, contamination `2/50`, length ratio `1.12`, and title-suffix preservation `22/27`.
- `final_gap04_multitask_robustwpo` also reaches reference-aware overall `2.22`, with the strongest title-style score among learned systems (`4.70`) and the same `5/50` exact matches.
- Relative to the looser gap-$0.2$ variants, the main gain is not higher judge overall, but much cleaner title behavior:
  - sigmoid: chrF++ `20.93 -> 30.01`, length ratio `1.74 -> 1.12`;
  - robust+wpo: contamination `7/50 -> 2/50`, length ratio `1.58 -> 1.26`.
- The subset effect is structured:
  - on title-suffix items, both gap-$0.4$ variants reach mean overall `2.41`, above `baseline3_2_multitask` (`2.15`) and above the gap-$0.2$ variants;
  - on non-suffix items, gap-$0.2$ sigmoid remains stronger (`2.30`), so strict filtering appears to trade breadth for cleaner canonical-title reconstruction.
- Implication: the new paper claim should be that structured multitask SFT is the right base, and aggressive pair filtering is the key step that turns DPO from an unstable add-on into a genuinely competitive method.

## [2026-04-10] Legacy UNK-base DPO remains unstable and should still be reported as a negative control
- `final_v2` is worse than its selected SFT base (`baseline3_1_unk`) on chrF++ (`19.39` vs `22.11`), exact match (`2/50` vs `3/50`), and contamination (`9/50` vs `3/50`).
- The legacy `final` model is even more unstable, with heavy output corruption and very poor PPL.
- Implication: the paper now has a stronger contrast: old DPO fails, but multitask-base gap-filtered DPO partially rescues the method.

## [2026-04-10] Deterministic title normalization reveals that robust DPO's main weakness is stopping behavior
- Applying `eval/clean_predictions.py --title-normalize` to `final_gap02_multitask_robustwpo` removes contamination from `7/50` to `0/50` and raises chrF++ from `27.67` to `29.06`.
- The same cleaning changes the fixed-key reference-aware judge only marginally (`2.12` overall before and after), which suggests that the robust model already had useful content but unstable surface realization.
- The same normalization helps legacy `final_v2` only modestly (`19.39` to `21.98` chrF++), so this is not a universal post-processing fix.
- Implication: the next engineering target is title-aware stopping / decoding control, not just stronger preference loss functions.

## [2026-04-10] DPO preference pairs contain non-trivial noise
- `data/dpo/dpo_pairs.jsonl` contains `3` duplicate chosen/rejected pairs and `3` non-finite reward entries.
- In a quick gold-proxy audit using `difflib.SequenceMatcher` against the original synthetic targets, reward-chosen outputs are closer to gold than reward-rejected outputs only `71.2%` of the time.
- For low reward-gap pairs (`<= 0.1`), the chosen-better rate drops to `62.3%`.
- Implication: a stricter reward-gap filter is a strong next experiment.

## [2026-04-10] The task should be described as short-text / title translation
- The real Tangut corpus has only `491` items total, and the test set contains many short title-like strings.
- Average reference length in the test set is about `8` characters.
- Implication: avoid claiming general Tangut sentence translation.

---

# Engineering Findings

## [2026-04-10] Multitask outputs must be cleaned before fair scoring
- `baseline3_2_multitask` emits `<dict_match>` and `<literal>` scaffolding during inference.
- The repository already cleans these tags before running the official metric script.
- Decision: any future reference-aware evaluation must use the cleaned outputs, not the raw tagged text.

## [2026-04-10] Best-SFT selection for DPO currently excludes the strongest multitask model
- `run.sh` selects the DPO base from `mixed`, `unk`, and `semantic` only, using chrF as the selector.
- `multitask` is not included in the selector even though it is the strongest current SFT result.
- Decision: reviewers would have questioned this, and the follow-up confirmed that the omission mattered materially.

## [2026-04-10] The reference-aware judge was silently broken by a stale environment key
- `eval/reference_aware_judge.py` was reading `AZURE_OPENAI_API_KEY` from the environment first.
- The exported key in the current shell returns `401`, while the repository default key still works.
- Decision: the judge now defaults to the repository key unless an explicit override is passed, which restores reproducible fixed-key reruns.

## [2026-04-10] Existing assets are enough for paper drafting before new training
- The repository already contains:
  - all major baseline predictions,
  - metrics for every key variant,
  - merged checkpoints for the main SFT families,
  - a DPO checkpoint,
  - a human-reference anchor.
- Decision: write the paper from the current evidence first, then run only reviewer-facing follow-up experiments.
