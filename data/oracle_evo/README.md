# Oracle EVOBC Portability Assets

This directory stores the non-Tangut assets needed to instantiate the
symbol-grounded local-training stack on EVOBC oracle-bone IDs.

## Layout

- `assets/`: deterministic ID, surrogate, and metadata tables
- `dictionary/`: Tangut-style dictionary files generated from EVOBC
- `raw/`: hand-curated oracle probe inputs before surrogate encoding
- `eval/`: encoded oracle probes that match the Tangut JSONL format

## Important Constraint

The generated surrogate symbols are an implementation detail only. The auditable
external representation remains the EVOBC ID inventory. Keep those IDs in
`input_ids` or `metadata` whenever you curate real oracle examples.

## Current Status

- `assets/` and `dictionary/` were generated from `Key&Value.json`
- `sft/synthetic_multitask.jsonl` is the broad-coverage 50k multitask pseudo-parallel corpus
- `sft/synthetic_multitask_short24.jsonl` is the shorter 50k multitask corpus recommended for oracle short-text probes
- `raw/real_probe.template.jsonl` is only a format template
- `raw/obimd_canonicalized_{train,dev,test}.jsonl` stores a real OBIMD-derived oracle probe canonicalized into EVOBC IDs
- `eval/obimd_canonicalized_{train,dev,test}_encoded.jsonl` stores the surrogate-encoded form of that probe
- `checkpoints/oracle_evo_sft_short24/final` is a completed multitask SFT checkpoint trained on `sft/synthetic_multitask_short24.jsonl`
- `results/oracle_evo_probe/` stores first-pass real-probe results:
  - zero-shot Qwen2.5-7B: `0/200` exact, `0.44` chrF++
  - dictionary-grounded prompt: `40/200` exact, `46.20` chrF++
  - multitask SFT (`<literal>` readout): `29/200` exact, `40.56` chrF++
  - simple open 2-way selector (`hybrid_simple_2way/`): `44/200` exact,
    `47.81` chrF++, `0.5%` contamination

The portability result to keep in mind is workflow-level rather than
single-model: the dictionary-grounded prompt is the strongest standalone oracle
system, while the learned MT-SFT output supplies complementary short-form
repairs that a minimal open hybrid can recover.
