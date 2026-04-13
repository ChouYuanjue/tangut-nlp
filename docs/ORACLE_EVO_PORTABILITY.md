# EVOBC Portability Runbook

This runbook adds the minimum extra machinery needed to port the
dictionary-grounded local-training stack to oracle-bone IDs without touching
the Tangut mainline code.

## Scope

- Data source: EVOBC only
- Transfer target: symbol-grounded preprocessing, synthetic SFT, and gap-filtered DPO
- Out of scope: Tangut-specific title reranking, catalog adjudication, and title-style judging

The public EVOBC repository exposes external IDs such as `00001` rather than
Unicode oracle strings. The current Tangut code assumes that each source-side
symbol is a single Python character. The adapter therefore compiles each EVOBC
ID into a reversible single-codepoint surrogate from a private-use Unicode
range and emits Tangut-style dictionary assets.

## 1. Build EVOBC Assets

If the EVOBC repository is already cloned at `/tmp/character-Evolution-Dataset`,
the script will find it automatically.

```bash
cd /home/runnel/tangut-nlp
python3 scripts/build_evobc_portability_assets.py
```

This writes:

- `data/oracle_evo/assets/id_to_char.json`
- `data/oracle_evo/assets/char_to_ids.json`
- `data/oracle_evo/assets/id_to_surrogate.json`
- `data/oracle_evo/assets/surrogate_to_id.json`
- `data/oracle_evo/dictionary/evo_dictionary.json`
- `data/oracle_evo/dictionary/evo_reward_dict.json`

Use `--dry-run` first if you only want a summary.

## 2. Encode Real Oracle ID Sequences

If you manually curate oracle examples as JSONL with an `input` field
containing whitespace-delimited IDs, rewrite them into surrogate strings:

```bash
cd /home/runnel/tangut-nlp
python3 scripts/encode_oracle_jsonl.py \
  --input data/oracle_evo/raw/real_probe.jsonl \
  --output data/oracle_evo/eval/real_probe_encoded.jsonl
```

The script stores the canonicalized original IDs under `input_ids` by default.

## 2.5 Build the OBIMD Canonicalized Probe

The current repository snapshot now includes a small real oracle probe derived
from OBIMD sentence-level reading sequences and mapped into the EVOBC ID
inventory through shared modern-character labels.

```bash
cd /home/runnel/tangut-nlp
python3 scripts/build_obimd_oracle_probe.py \
  --obimd-root /tmp/OBIMD-hf \
  --output-root data/oracle_evo \
  --output-prefix obimd_canonicalized \
  --train-size 50 \
  --dev-size 50 \
  --test-size 200 \
  --seed 42
```

This writes:

- `data/oracle_evo/raw/obimd_canonicalized_train.jsonl`
- `data/oracle_evo/raw/obimd_canonicalized_dev.jsonl`
- `data/oracle_evo/raw/obimd_canonicalized_test.jsonl`
- `data/oracle_evo/eval/obimd_canonicalized_train_encoded.jsonl`
- `data/oracle_evo/eval/obimd_canonicalized_dev_encoded.jsonl`
- `data/oracle_evo/eval/obimd_canonicalized_test_encoded.jsonl`
- `data/oracle_evo/raw/obimd_canonicalized_summary.json`

The held-out paper-facing split is the 200-item
`obimd_canonicalized_test_encoded.jsonl`.

## 3. Generate Synthetic Oracle SFT Data

Reuse the existing Tangut synthesizer with the EVOBC dictionary.

Broad-coverage baseline:

```bash
cd /home/runnel/tangut-nlp
python3 src/data_synthesis.py \
  --dictionary-path data/oracle_evo/dictionary/evo_dictionary.json \
  --output data/oracle_evo/sft/synthetic_multitask.jsonl \
  --mode multitask \
  --max-samples 50000 \
  --seed 42
```

Short-text-oriented variant recommended for oracle portability:

```bash
cd /home/runnel/tangut-nlp
python3 src/data_synthesis.py \
  --dictionary-path data/oracle_evo/dictionary/evo_dictionary.json \
  --output data/oracle_evo/sft/synthetic_multitask_short24.jsonl \
  --mode multitask \
  --max-samples 50000 \
  --min-length 4 \
  --max-length 24 \
  --seed 42
```

In the current repository snapshot both corpora are already prepared:

- `data/oracle_evo/sft/synthetic_multitask.jsonl`: broader coverage, mean input length about `24.9`
- `data/oracle_evo/sft/synthetic_multitask_short24.jsonl`: shorter probe-oriented variant, mean input length about `16.3`

`mixed`, `unk`, and `semantic` remain available, but the fastest portability
path is still `multitask`, and the short24 variant is the better default for a
small oracle short-text probe.

## 4. Combine with Real Oracle Pairs

If you collect a small real oracle probe, keep the file format aligned with the
Tangut train split:

```json
{"instruction": "请将以下甲骨文符号序列翻译为现代中文：", "input": "<surrogate string>", "output": "<short Chinese reference>"}
```

Then combine it with the synthetic set using the existing combiner:

```bash
cd /home/runnel/tangut-nlp
python3 src/combine_data.py \
  --synthetic data/oracle_evo/sft/synthetic_multitask_short24.jsonl \
  --real data/oracle_evo/sft/real_train.jsonl \
  --output data/oracle_evo/sft/combined_multitask.jsonl \
  --upsample-real 10
```

## 5. Train SFT and DPO

The existing training scripts can then be reused directly by swapping the data
paths:

```bash
cd /home/runnel/tangut-nlp
python3 experiments/baseline3_synthetic_sft.py \
  --train-data data/oracle_evo/sft/combined_multitask.jsonl \
  --output-dir checkpoints/oracle_evo_sft
```

```bash
cd /home/runnel/tangut-nlp
python3 experiments/generate_candidates.py \
  --sft-model checkpoints/oracle_evo_sft/final \
  --train-data data/oracle_evo/sft/combined_multitask.jsonl \
  --reward-dict data/oracle_evo/dictionary/evo_reward_dict.json \
  --output data/oracle_evo/dpo/dpo_pairs.jsonl
python3 scripts/filter_dpo_pairs.py \
  --input data/oracle_evo/dpo/dpo_pairs.jsonl \
  --output data/oracle_evo/dpo/dpo_pairs_gap04.jsonl \
  --min-gap 0.4 \
  --dedupe
python3 experiments/final_dpo.py \
  --dpo-data data/oracle_evo/dpo/dpo_pairs_gap04.jsonl \
  --sft-model checkpoints/oracle_evo_sft/final \
  --output-dir checkpoints/oracle_evo_dpo
```

## 6. What Not to Reuse Blindly

Do not present the following components as script-agnostic:

- `experiments/open_hybrid_heuristic.py`
- `experiments/open_local_reranker.py`
- `eval/reference_aware_judge.py`

These components encode Tangut title-specific priors and should be rewritten
before any oracle-bone workflow claims are made.

## 7. Runtime Expectations

These estimates are grounded in the current repository's Tangut runs with the
same Qwen2.5-7B LoRA code path and should be treated as planning numbers rather
than oracle-specific measurements.

- Multitask SFT: the existing Tangut run used `54,000` rows, `3` epochs,
  `--batch-size 4`, `--grad-accum 4`, and `accelerate --num_processes 2`. That
  run finished in about `4h08m` (`14,899s`). An oracle run with a combined set
  of about `51k` to `55k` rows should therefore land around `3.9h` to `4.2h`
  on the same `2xA100 80GB` setup. On a single A100, expect roughly double.
- DPO candidate generation: the existing Tangut stage with `--max-inputs 5000`
  and `N=5` candidates took about `10m31s` on `2xA100` and produced `3,867`
  usable pairs. The actual text generation was fast; most wall-clock time came
  from reward scoring, especially perplexity.
- DPO training: the existing Tangut single-GPU DPO run trained `3,113` pairs
  for `780` optimization steps in about `38.1m` on `1xA100`. Rough planning
  numbers are therefore about `6m` for `500` pairs, `12m` for `1,000` pairs,
  and `24m` for `2,000` pairs.

For a first oracle probe, SFT is the main cost. DPO training itself is cheap;
the expensive part of a full DPO cycle is pair generation plus scoring.

## 8. Lowest-Cost Defensible Validation

If the goal is to answer a reviewer quickly with something valid, the best
trade-off is a small real oracle probe centered on multitask SFT, not on the
full Tangut workflow.

Recommended minimum:

1. Curate `100` to `300` real oracle short pairs from a single consistent
   source and keep them in the existing JSONL format.
2. Split them into a real held-out test set first. If possible, reserve at
   least `100` test examples. Use the remainder for a tiny train/dev split only
   if you want to upsample a few real examples into SFT.
3. Run two primary systems on the held-out real probe:
   - frontier dictionary-grounded prompting
   - `multitask` SFT trained on `synthetic_multitask_short24.jsonl`, optionally
     plus a very small real-train upsample
4. If the two systems show complementary errors, add only a minimal open
   2-way selector before considering any oracle-side DPO. In the current
   repository snapshot this is already enough to turn portability into a
   positive workflow result.
5. Report only real-set metrics in the rebuttal table. Treat synthetic data as
   training infrastructure, not as evaluation evidence.

This is the smallest result that still answers the portability question:
whether the symbol-grounded synthetic-training recipe transfers beyond Tangut.

The default oracle-first choice should be:

- synthetic source: `data/oracle_evo/sft/synthetic_multitask_short24.jsonl`
- model family: `multitask` only
- no oracle DPO unless the SFT run already shows a clear held-out gain or a
  clear complementary error profile

If GPU time is extremely limited, skip DPO entirely for the first response.
Reviewer value per GPU-hour is much higher for "frontier vs multitask SFT on a
real oracle probe" than for adding a small DPO row immediately.

Current snapshot:

- `oracle_evo_sft_short24` SFT finished successfully in about `2h26m`
- held-out real oracle probe: `data/oracle_evo/eval/obimd_canonicalized_test_encoded.jsonl`
- zero-shot Qwen2.5-7B: `0/200` exact, `0.44` chrF++
- dictionary-grounded prompt: `40/200` exact, `46.20` chrF++
- multitask SFT (`<literal>` field readout): `29/200` exact, `40.56` chrF++
- simple open 2-way selector over dict prompt + MT-SFT: `44/200` exact,
  `47.81` chrF++, `0.5%` contamination

The current best portability narrative is therefore not ``local SFT beats
prompting.'' It is ``dictionary-grounded prompting is the strongest single
oracle model, while MT-SFT contributes complementary repairs that a minimal
open hybrid can recover.'' That keeps Tangut as the center and uses oracle only
as a conservative cross-script stress test of the workflow idea.

## 9. Main Reviewer Attack Surfaces

The current portability story is much cleaner than before, but the following
attack points remain live even after the first small real oracle result.

- Canonicalized rather than fully matched oracle task. The repository now has a
  held-out real oracle table, so "no empirical transfer result" is no longer
  the main attack. The live objection is instead that the current probe is
  canonicalized OBIMD$\rightarrow$EVOBC decipherment, not a matched
  oracle-to-modern-Chinese short-title benchmark.
- Synthetic distribution mismatch. Classical Chinese replacement is a fast way
  to build a pseudo-parallel set, but oracle inscriptions are shorter and more
  formulaic. This is why `short24` should be the default run and why the paper
  should not oversell the broad `50k` corpus as historically faithful.
- Overclaiming full workflow transfer. The adapter does not validate the
  Tangut-specific selector, reranker, or reference-aware judge on oracle data.
  Keep the claim limited to the symbol-grounded preprocessing plus local
  training stack.
- Small-sample instability. If the oracle probe is small, exact match and judge
  gains can be noisy. Prefer chrF++ plus exact match, and if time allows add a
  paired bootstrap or at least confidence intervals on exact match.
- Reward mismatch for DPO. The reward is still built from lexical coverage plus
  a weak perplexity tie-breaker. On oracle data, a weak or noisy reward can be
  attacked more easily than on Tangut, so DPO should remain secondary until the
  SFT probe is established.

## 10. Narrative Guardrails

The safest rebuttal wording is:

- claim portability of the symbolic interface and local-training recipe
- show one small real oracle probe as evidence that the transfer is empirical
  rather than merely infrastructural
- avoid claiming that Tangut title-style workflow modules transfer unchanged

In other words, keep the paper's center fixed. The oracle addition should
function as a narrow external validation of the method family, not as an
attempt to turn the paper into a second benchmark paper.
