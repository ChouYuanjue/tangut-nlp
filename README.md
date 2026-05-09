# Auditable Historical-Script Interpretation

This repository supports the Culture x AI workshop paper:

**Evidential Humility as a Cultural Value for AI: Auditable Workflows for Historical-Script Interpretation**

The project studies generative AI for historical-script translation as an interpretive cultural technology, not only as a low-resource machine-translation benchmark. The central question is: what should count as success when AI is used to support interpretation of fragile cultural evidence?

Our answer is organized around two positive cultural values:

- **Evidential humility**: preserve uncertainty, avoid unsupported expansion, and prevent model or prompt artifacts from entering a catalog-like answer.
- **Interpretive reversibility**: keep candidate provenance, selector rationale, and audit records so a scholar can inspect and contest the workflow.

The repository is intended to make the paper's workflow auditable. It contains scripts for generating deterministic diagnostics, candidate-selection comparisons, and small portability probes from existing prediction artifacts.

## What This Repository Is Not

This is not a claim of state-of-the-art historical-script translation.

This is not a deployment-ready cataloging system.

This is not a paper about replacing expert readers with a single fluent model output.

The paper treats models as producers of candidate interpretations. The accountable unit is the workflow: historical-script item plus dictionary evidence, candidate interpretations, value-aware selection, audit record, and human review.

## Core Evaluation Ideas

Standard metrics remain useful but are not sufficient. The Culture x AI framing adds diagnostics that ask whether a workflow preserves the conditions for later human interpretation.

The main diagnostics are:

- **contamination**: Latin fragments, prompt debris, `[UNK]`, or other artifacts in outputs;
- **truncation**: collapse of a compact title into an underspecified fragment;
- **over-expansion**: unsupported length growth relative to the catalog-style reference;
- **title-form and suffix errors**: loss of compact catalog-title conventions;
- **numeral errors**: mismatch in canonical numeral information;
- **unsupported narrativization**: conversion of a compact title/gloss into a modern sentence with unsupported subjects, predicates, punctuation, or narrative relations;
- **switch count**: how often a selector departs from the frontier candidate;
- **trace rows / audit records**: whether candidate provenance and selector rationale are preserved.

These are transparent proxy measures. They do not replace expert humanities judgment; they make culturally relevant failure modes inspectable.

## Main Workflow

The Tangut workflow uses a 50-item held-out short-title set and compares:

- a dictionary-grounded frontier prompt;
- local SFT candidates;
- an UNK-preserving local branch;
- an auxiliary DPO candidate used only for diversity;
- deterministic selectors over existing candidate strings;
- a closed adjudicator reported only as headroom.

The workflow emphasizes conservative selection. Local models are not treated as replacements for frontier models. They are alternative witnesses that sometimes repair truncation, expose uncertainty, or preserve compact form.

## Key Scripts

- `scripts/culture_ai_diagnostics.py`
  Generates the Culture x AI diagnostic tables, selector ablation, per-item diagnostic CSV, and qualitative-case shortlist.
- `scripts/eval_oracle_portability.py`
  Evaluates the 200-item oracle-bone portability probe.
- `scripts/oracle_two_way_selector.py`
  Builds the conservative two-way oracle-bone selector over dictionary-prompt and SFT-literal outputs.
- `scripts/build_obimd_oracle_probe.py`
  Constructs the oracle-bone portability probe from OBIMD/EVOBC-derived assets.

## Reproducing the Culture x AI Diagnostics

From the repository root:

```bash
python3 scripts/culture_ai_diagnostics.py
```

This command uses existing prediction files and does not call external models. It writes:

- `results/culture_ai_diagnostics/table_a_standard_metrics.csv`
- `results/culture_ai_diagnostics/table_b_cultural_metrics.csv`
- `results/culture_ai_diagnostics/selector_value_ablation.csv`
- `results/culture_ai_diagnostics/per_item_diagnostics.csv`
- `results/culture_ai_diagnostics/qualitative_cases.csv`

The per-item diagnostic file is the primary audit artifact. It records the source, reference, frontier/local/DPO candidates, selector output, chosen method, standard metrics, cultural diagnostic flags, selector action, switch reason, and audit-needed flag.

## Oracle-Bone Portability Probe

The oracle-bone experiment is a portability stress test for the value-aware workflow. It is not a second matched short-title benchmark.

The probe uses a 200-item canonicalized OBIMD-to-EVOBC dataset. Dictionary-grounded prompting is the strongest single model in the reported setup, while a simple open two-way selector can reduce contamination by choosing a clean SFT-literal alternative when the dictionary prompt produces artifacts.

## Repository Layout

- `data/dictionary/`: Tangut dictionary assets used by prompting and diagnostics.
- `data/eval/`: Tangut evaluation splits.
- `experiments/`: generation, training, and selector experiments.
- `eval/`: metric computation and lexical-coverage utilities.
- `scripts/`: artifact generation, diagnostic, oracle-bone, and evaluation scripts.
- `src/`: data preparation, dictionary handling, prompt construction, and synthesis helpers.
- `configs/`: training configuration files.
- `paper_icml_culture_ai_2026/`: Culture x AI paper source and build artifacts in the working tree.

Generated predictions, checkpoints, large training corpora, paper build outputs, and legacy paper directories are excluded from the anonymous repository view.

## Environment

The original working environment used:

```bash
conda activate tangut-nlp
pip install -r requirements.txt
```

Some historical evaluation scripts can use external judge APIs, but the Culture x AI diagnostic script is offline and deterministic.

If running API-backed scripts, configure credentials explicitly through environment variables; no credentials are stored in this repository.

## Paper Framing

The paper's message is not "we built a better hybrid translator." It is:

**Generative AI for historical scripts should be evaluated as an accountable interpretive cultural technology.**

Success is not only exact match or chrF++. It is also the ability to preserve catalog form, expose uncertainty, avoid unsupported cultural normalization, maintain provenance, and support scholar-facing review.
