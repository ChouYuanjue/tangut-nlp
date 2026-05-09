# Supplementary Data and Code

This directory contains data and code artifacts for:

`Evidential Humility as a Cultural Value for AI: Auditable Workflows for Historical-Script Interpretation`

The appendix is included in the main paper PDF after the references. It is not submitted as separate supplementary material.

## Contents

- `artifacts/table_a_standard_metrics.csv`: standard Tangut metrics.
- `artifacts/table_b_cultural_metrics.csv`: cultural/interpretive diagnostics.
- `artifacts/selector_value_ablation.csv`: selector value ablation.
- `artifacts/per_item_diagnostics.csv`: required per-item diagnostic CSV.
- `artifacts/qualitative_cases.csv`: qualitative case shortlist.
- `scripts/culture_ai_diagnostics.py`: offline deterministic diagnostic generator.

## Reproduce

From the repository root:

```bash
python3 scripts/culture_ai_diagnostics.py
```

The diagnostic script uses existing prediction files and does not call external models.
