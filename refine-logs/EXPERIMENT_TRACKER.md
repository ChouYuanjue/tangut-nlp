# Experiment Tracker

| ID | Experiment | Status | Priority | Notes |
|----|------------|--------|----------|-------|
| M0.1 | Reference-aligned diagnostics from existing predictions | DONE | MUST-RUN | Derived exact match, contamination, length ratio, suffix preservation |
| M0.2 | Human-reference anchor analysis | DONE | MUST-RUN | Confirms metric misalignment |
| M0.3 | DPO pair-quality audit | DONE | MUST-RUN | Quick audit completed from existing pairs |
| M1.1 | Reference-aware judge rerun | DONE | MUST-RUN | Real Azure API run completed; `results/reference_eval_suite/summary.csv` written |
| M1.2 | Subset analysis by title suffix / length | DONE | SHOULD-RUN | `scripts/analyze_reference_eval_subsets.py` generated reviewer-facing subset summaries |
| M2.1 | Gap-filtered DPO on UNK base | READY | SHOULD-RUN | `data/dpo/dpo_pairs_gap02.jsonl` and `data/dpo/dpo_pairs_gap04.jsonl` prepared |
| M2.2 | DPO from multitask base | RUNNING | SHOULD-RUN | `screen`: `dpo_mt_sigmoid_gap02`; auto-eval waiter: `eval_mt_sigmoid_gap02` |
| M2.3 | Robust+weighted DPO from multitask base | RUNNING | SHOULD-RUN | Literature-backed noise/off-policy control; `screen`: `dpo_mt_robustwpo_gap02` |
| M3.1 | Semantic-projection cleanup | DEFER | NICE-TO-HAVE | Only if reviewers demand it |
