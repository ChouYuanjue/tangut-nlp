#!/usr/bin/env python3
"""Targeted robustness study for the open Tangut selector.

This script supports two reviewer-facing supplemental analyses that directly
serve the paper's main narrative:

1. Pool ablation: how much complementarity is recovered when we add local
   candidates one by one to the frontier baseline.
2. Sensitivity: how stable the open selector is to small changes in its priors
   and guarded-switch thresholds.

The outputs are lightweight, fully offline, and reproducible from existing
prediction files.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

import sacrebleu

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.open_hybrid_heuristic import (  # noqa: E402
    BAD_PATTERN,
    candidate_score,
    choose_guarded,
    load_jsonl,
)


DIAG_TITLE_SUFFIXES = set("經論記疏頌儀義傳錄贊序字品")

DEFAULT_PREDICTIONS = {
    "frontier": REPO_ROOT / "results/frontier_deepseek_v32_fewshot_cot/predictions.jsonl",
    "unk": REPO_ROOT / "results/baseline3_1_unk/predictions.jsonl",
    "mtsft": REPO_ROOT / "results/baseline3_2_multitask/predictions_cleaned.jsonl",
    "dpo": REPO_ROOT / "results/final_gap04_multitask_sigmoid/predictions.jsonl",
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def corpus_chrf(hyps: list[str], refs: list[str]) -> float:
    return sacrebleu.corpus_chrf(hyps, [refs], char_order=6, word_order=2, beta=2).score


def build_selector_rows(
    *,
    rows_by_name: dict[str, list[dict]],
    candidate_names: list[str],
    frontier_name: str,
    mode: str,
    prefix_weight: float,
    suffix_weight: float,
    title_suffix_bonus: float,
    length_penalty_weight: float,
    contamination_penalty: float,
    too_short_penalty: float,
    too_long_penalty: float,
    frontier_prior: float,
    switch_margin: float,
    switch_consensus_delta: int,
) -> list[dict]:
    num_rows = len(rows_by_name[candidate_names[0]])
    output_rows = []

    for idx in range(num_rows):
        candidates = []
        source = None
        reference = None
        glosses = None
        for name in candidate_names:
            row = rows_by_name[name][idx]
            if source is None:
                source = row["input"]
                reference = row.get("reference", "")
                glosses = row.get("glosses", "")
            elif row["input"] != source:
                raise ValueError(f"Input mismatch at row {idx + 1}: {name}")
            pred = (row.get("prediction", "") or "").strip()
            if not pred:
                continue
            candidates.append({"name": name, "prediction": pred})

        if not candidates:
            raise ValueError(f"No non-empty candidates at row {idx + 1}")

        preds = [c["prediction"] for c in candidates]
        median_length = sorted(len(x) for x in preds)[len(preds) // 2]
        scored = []
        for cand in candidates:
            others = [x for x in preds if x != cand["prediction"]]
            score, diagnostics = candidate_score(
                pred=cand["prediction"],
                others=others,
                source_length=len(source or ""),
                median_length=median_length,
                is_frontier=(cand["name"] == frontier_name),
                prefix_weight=prefix_weight,
                suffix_weight=suffix_weight,
                title_suffix_bonus=title_suffix_bonus,
                length_penalty_weight=length_penalty_weight,
                contamination_penalty=contamination_penalty,
                too_short_penalty=too_short_penalty,
                too_long_penalty=too_long_penalty,
                frontier_prior=frontier_prior,
            )
            scored.append(
                {
                    "name": cand["name"],
                    "prediction": cand["prediction"],
                    "score": score,
                    "diagnostics": diagnostics,
                }
            )

        if mode == "guarded":
            best = choose_guarded(
                scored,
                frontier_name,
                switch_margin=switch_margin,
                switch_consensus_delta=switch_consensus_delta,
            )
        else:
            best = max(scored, key=lambda x: (x["score"], x["name"] == frontier_name))

        output_rows.append(
            {
                "input": source,
                "reference": reference,
                "prediction": best["prediction"],
                "glosses": glosses,
                "selector_basis": best["name"],
                "selector_score": round(best["score"], 4),
                "selector_mode": mode,
                "selector_diagnostics": best["diagnostics"],
                "candidate_scores": [
                    {
                        "name": item["name"],
                        "score": round(item["score"], 4),
                        **item["diagnostics"],
                    }
                    for item in scored
                ],
            }
        )

    return output_rows


def evaluate_predictions(rows: list[dict], frontier_name: str) -> OrderedDict:
    refs = [row["reference"] for row in rows]
    hyps = [row["prediction"] for row in rows]
    exact = sum(int(h == r) for h, r in zip(hyps, refs))
    contamination = sum(int(bool(BAD_PATTERN.search(h))) for h in hyps)

    suffix_total = 0
    suffix_ok = 0
    for hyp, ref in zip(hyps, refs):
        ref_suffixes = {ch for ch in ref if ch in DIAG_TITLE_SUFFIXES}
        if not ref_suffixes:
            continue
        suffix_total += 1
        if all(ch in hyp for ch in ref_suffixes):
            suffix_ok += 1

    basis_counts: dict[str, int] = {}
    switched_from_frontier = 0
    for row in rows:
        basis = row["selector_basis"]
        basis_counts[basis] = basis_counts.get(basis, 0) + 1
        if basis != frontier_name:
            switched_from_frontier += 1

    return OrderedDict(
        [
            ("num_examples", len(rows)),
            ("corpus_chrf", round(corpus_chrf(hyps, refs), 4)),
            ("exact_match", exact),
            ("exact_match_rate", round(exact / len(rows), 4)),
            ("contamination_rate", round(contamination / len(rows), 4)),
            (
                "length_ratio",
                round(
                    sum(len(h) for h in hyps) / max(1, sum(len(r) for r in refs)),
                    4,
                ),
            ),
            ("title_suffix_ok", suffix_ok),
            ("title_suffix_total", suffix_total),
            ("switched_from_frontier", switched_from_frontier),
            ("basis_counts", json.dumps(basis_counts, ensure_ascii=False, sort_keys=True)),
        ]
    )


def pool_ablation_configs() -> list[dict]:
    return [
        {"name": "frontier_only", "candidates": ["frontier"], "mode": "score_max"},
        {"name": "frontier_plus_unk", "candidates": ["frontier", "unk"], "mode": "guarded"},
        {"name": "frontier_plus_mtsft", "candidates": ["frontier", "mtsft"], "mode": "guarded"},
        {"name": "frontier_plus_dpo", "candidates": ["frontier", "dpo"], "mode": "guarded"},
        {"name": "frontier_plus_unk_dpo", "candidates": ["frontier", "unk", "dpo"], "mode": "guarded"},
    ]


def sensitivity_configs() -> list[dict]:
    defaults = {
        "prefix_weight": 1.4,
        "suffix_weight": 1.2,
        "title_suffix_bonus": 2.5,
        "length_penalty_weight": 1.0,
        "contamination_penalty": 6.0,
        "too_short_penalty": 4.0,
        "too_long_penalty": 2.0,
        "frontier_prior": 0.75,
        "switch_margin": 3.0,
        "switch_consensus_delta": 2,
    }
    configs = [
        {"name": "default_guarded", "mode": "guarded", **defaults},
        {"name": "score_max", "mode": "score_max", **defaults},
        {"name": "frontier_prior_0.0", "mode": "guarded", **defaults, "frontier_prior": 0.0},
        {"name": "frontier_prior_0.25", "mode": "guarded", **defaults, "frontier_prior": 0.25},
        {"name": "frontier_prior_1.5", "mode": "guarded", **defaults, "frontier_prior": 1.5},
        {"name": "switch_margin_2.0", "mode": "guarded", **defaults, "switch_margin": 2.0},
        {"name": "switch_margin_4.0", "mode": "guarded", **defaults, "switch_margin": 4.0},
        {"name": "title_bonus_1.5", "mode": "guarded", **defaults, "title_suffix_bonus": 1.5},
        {"name": "title_bonus_3.5", "mode": "guarded", **defaults, "title_suffix_bonus": 3.5},
    ]
    return configs


def load_prediction_map(overrides: list[str]) -> dict[str, Path]:
    pred_map = dict(DEFAULT_PREDICTIONS)
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=PATH, got {spec}")
        name, raw_path = spec.split("=", 1)
        pred_map[name.strip()] = Path(raw_path).expanduser()
    return pred_map


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pool-ablation and sensitivity studies for the open selector.")
    parser.add_argument(
        "--pred",
        action="append",
        default=[],
        help="Optional NAME=PATH override for prediction files.",
    )
    parser.add_argument(
        "--study",
        choices=["pool_ablation", "sensitivity", "all"],
        default="all",
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results/analysis/open_selector_study"),
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predictions.jsonl for each studied variant.",
    )
    parser.add_argument("--frontier-name", default="frontier")
    args = parser.parse_args()

    pred_map = load_prediction_map(args.pred)
    rows_by_name = {name: load_jsonl(path) for name, path in pred_map.items()}

    lengths = {len(rows) for rows in rows_by_name.values()}
    if len(lengths) != 1:
        raise ValueError(f"Prediction length mismatch: {lengths}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    studies: list[tuple[str, list[dict]]] = []
    if args.study in {"pool_ablation", "all"}:
        studies.append(("pool_ablation", pool_ablation_configs()))
    if args.study in {"sensitivity", "all"}:
        studies.append(("sensitivity", sensitivity_configs()))

    for study_name, configs in studies:
        for config in configs:
            candidate_names = config.get("candidates", ["frontier", "unk", "dpo"])
            selector_rows = build_selector_rows(
                rows_by_name=rows_by_name,
                candidate_names=candidate_names,
                frontier_name=args.frontier_name,
                mode=config["mode"],
                prefix_weight=config.get("prefix_weight", 1.4),
                suffix_weight=config.get("suffix_weight", 1.2),
                title_suffix_bonus=config.get("title_suffix_bonus", 2.5),
                length_penalty_weight=config.get("length_penalty_weight", 1.0),
                contamination_penalty=config.get("contamination_penalty", 6.0),
                too_short_penalty=config.get("too_short_penalty", 4.0),
                too_long_penalty=config.get("too_long_penalty", 2.0),
                frontier_prior=config.get("frontier_prior", 0.75),
                switch_margin=config.get("switch_margin", 3.0),
                switch_consensus_delta=config.get("switch_consensus_delta", 2),
            )
            metrics = evaluate_predictions(selector_rows, args.frontier_name)
            row = OrderedDict(
                [
                    ("study", study_name),
                    ("variant", config["name"]),
                    ("candidate_pool", "+".join(candidate_names)),
                    ("mode", config["mode"]),
                    ("frontier_prior", config.get("frontier_prior", 0.75)),
                    ("title_suffix_bonus", config.get("title_suffix_bonus", 2.5)),
                    ("switch_margin", config.get("switch_margin", 3.0)),
                    ("switch_consensus_delta", config.get("switch_consensus_delta", 2)),
                ]
            )
            row.update(metrics)
            summary_rows.append(row)

            if args.save_predictions:
                pred_dir = output_dir / "predictions" / study_name
                pred_dir.mkdir(parents=True, exist_ok=True)
                pred_path = pred_dir / f"{config['name']}.jsonl"
                with pred_path.open("w", encoding="utf-8") as f:
                    for item in selector_rows:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    summary_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(summary_csv, summary_rows)

    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
