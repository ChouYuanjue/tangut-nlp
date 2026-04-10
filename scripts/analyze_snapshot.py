#!/usr/bin/env python3
"""Derive paper-facing diagnostics from stored Tangut-NLP artifacts.

This script computes:
1. Reference-aligned diagnostics from saved predictions:
   - exact match
   - contamination rate
   - average output length
   - length ratio vs reference
   - title-suffix preservation
2. A lightweight DPO-pair audit:
   - duplicate chosen/rejected pairs
   - non-finite rewards
   - reward-gap histogram
   - similarity-to-gold proxy via SequenceMatcher
   - similarity-to-gold proxy by reward-gap bin

The goal is to make paper-writing numbers reproducible from repository state.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import math
import re
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List


TITLE_SUFFIXES = set("經論記疏頌儀義傳錄贊序字品")
BAD_PATTERN = re.compile(
    r"[A-Za-z<>]|\[UNK\]|assistant|manuals|networks|shows|grat|dresses",
    re.IGNORECASE,
)


def load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def compute_reference_diagnostics(refs: List[str], preds: List[str]) -> dict:
    exact = sum(pred == ref for pred, ref in zip(preds, refs))
    contam = sum(bool(BAD_PATTERN.search(pred)) for pred in preds)

    ref_avg = (sum(len(ref) for ref in refs) / len(refs)) if refs else 0.0
    pred_avg = (sum(len(pred) for pred in preds) / len(preds)) if preds else 0.0

    suffix_total = 0
    suffix_ok = 0
    for pred, ref in zip(preds, refs):
        ref_suffixes = {ch for ch in ref if ch in TITLE_SUFFIXES}
        if not ref_suffixes:
            continue
        suffix_total += 1
        if ref_suffixes.issubset(set(pred)):
            suffix_ok += 1

    n = len(preds) if preds else 1
    return {
        "num_examples": len(preds),
        "exact_match": exact,
        "exact_match_rate": exact / n,
        "contamination_count": contam,
        "contamination_rate": contam / n,
        "avg_prediction_length": pred_avg,
        "avg_reference_length": ref_avg,
        "length_ratio": (pred_avg / ref_avg) if ref_avg else None,
        "title_suffix_ok": suffix_ok,
        "title_suffix_total": suffix_total,
        "title_suffix_rate": (suffix_ok / suffix_total) if suffix_total else None,
    }


def build_train_lookup(paths: Iterable[Path]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in load_jsonl(path):
            key = f"{row['instruction']}\n{row['input']}"
            lookup[key] = row["output"]
    return lookup


def similarity_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def make_gap_bin_bucket() -> dict:
    return {
        "count": 0,
        "gold_proxy_found": 0,
        "chosen_better": 0,
        "rejected_better": 0,
        "equal": 0,
    }


def gap_bin_name(gap: float) -> str:
    if gap <= 0.05:
        return "le_0.05"
    if gap <= 0.10:
        return "0.05_0.10"
    if gap <= 0.20:
        return "0.10_0.20"
    if gap <= 0.40:
        return "0.20_0.40"
    return "ge_0.40"


def audit_dpo_pairs(pairs: List[dict], train_lookup: Dict[str, str]) -> dict:
    duplicate_pairs = 0
    nonfinite_rewards = 0
    found = 0
    chosen_better = 0
    rejected_better = 0
    equal = 0
    gaps: List[float] = []
    low_gap_total = 0
    low_gap_good = 0

    gap_threshold_counts = OrderedDict(
        [
            ("ge_0.05", 0),
            ("ge_0.10", 0),
            ("ge_0.20", 0),
            ("ge_0.30", 0),
            ("ge_0.40", 0),
        ]
    )
    gap_bin_stats = OrderedDict(
        [
            ("le_0.05", make_gap_bin_bucket()),
            ("0.05_0.10", make_gap_bin_bucket()),
            ("0.10_0.20", make_gap_bin_bucket()),
            ("0.20_0.40", make_gap_bin_bucket()),
            ("ge_0.40", make_gap_bin_bucket()),
        ]
    )

    for row in pairs:
        if row.get("chosen") == row.get("rejected"):
            duplicate_pairs += 1

        chosen_reward = row.get("chosen_reward")
        rejected_reward = row.get("rejected_reward")
        if not math.isfinite(chosen_reward) or not math.isfinite(rejected_reward):
            nonfinite_rewards += 1
            continue

        gap = chosen_reward - rejected_reward
        gaps.append(gap)
        gap_bin_stats[gap_bin_name(gap)]["count"] += 1
        for key, threshold in (
            ("ge_0.05", 0.05),
            ("ge_0.10", 0.10),
            ("ge_0.20", 0.20),
            ("ge_0.30", 0.30),
            ("ge_0.40", 0.40),
        ):
            if gap >= threshold:
                gap_threshold_counts[key] += 1

        gold = train_lookup.get(row.get("prompt", ""))
        if gold is None:
            continue

        found += 1
        gap_bucket = gap_bin_stats[gap_bin_name(gap)]
        gap_bucket["gold_proxy_found"] += 1
        chosen_score = similarity_ratio(row["chosen"], gold)
        rejected_score = similarity_ratio(row["rejected"], gold)
        if chosen_score > rejected_score:
            chosen_better += 1
            gap_bucket["chosen_better"] += 1
        elif rejected_score > chosen_score:
            rejected_better += 1
            gap_bucket["rejected_better"] += 1
        else:
            equal += 1
            gap_bucket["equal"] += 1

        if gap <= 0.10:
            low_gap_total += 1
            if chosen_score > rejected_score:
                low_gap_good += 1

    mean_gap = (sum(gaps) / len(gaps)) if gaps else None
    min_gap = min(gaps) if gaps else None
    max_gap = max(gaps) if gaps else None
    gap_bin_quality = OrderedDict()
    for name, bucket in gap_bin_stats.items():
        denom = bucket["gold_proxy_found"]
        gap_bin_quality[name] = {
            **bucket,
            "chosen_better_rate": (bucket["chosen_better"] / denom) if denom else None,
        }

    return {
        "num_pairs": len(pairs),
        "duplicate_pairs": duplicate_pairs,
        "nonfinite_reward_pairs": nonfinite_rewards,
        "mean_reward_gap": mean_gap,
        "min_reward_gap": min_gap,
        "max_reward_gap": max_gap,
        "gold_proxy_found": found,
        "chosen_better_than_rejected": chosen_better,
        "rejected_better_than_chosen": rejected_better,
        "equal_proxy_similarity": equal,
        "chosen_better_rate": (chosen_better / found) if found else None,
        "low_gap_total": low_gap_total,
        "low_gap_chosen_better": low_gap_good,
        "low_gap_chosen_better_rate": (low_gap_good / low_gap_total) if low_gap_total else None,
        "gap_threshold_counts": gap_threshold_counts,
        "gap_bin_proxy_quality": gap_bin_quality,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Tangut-NLP repository snapshot.")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--test-set", default="data/eval/test_set.jsonl")
    parser.add_argument("--dpo-pairs", default="data/dpo/dpo_pairs.jsonl")
    parser.add_argument(
        "--train-data",
        nargs="*",
        default=[
            "data/sft/combined_sft_unk.jsonl",
            "data/sft/combined_sft.jsonl",
            "data/sft/combined_sft_semantic.jsonl",
        ],
        help="Train-data files used for the gold-proxy DPO audit.",
    )
    parser.add_argument(
        "--output-prefix",
        default="results/paper_diagnostics",
        help="Prefix for JSON/CSV outputs. Two files will be created: *_reference and *_dpo.",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="Only print diagnostics to stdout; do not write files.",
    )
    args = parser.parse_args()

    refs = [row["output"] for row in load_jsonl(Path(args.test_set))]

    prediction_files = OrderedDict(
        [
            ("baseline1", Path(args.results_dir) / "baseline1" / "predictions.jsonl"),
            ("baseline2", Path(args.results_dir) / "baseline2" / "predictions.jsonl"),
            ("baseline2_1_cot", Path(args.results_dir) / "baseline2_1_cot" / "predictions.jsonl"),
            ("baseline3", Path(args.results_dir) / "baseline3" / "predictions.jsonl"),
            ("baseline3_1_unk", Path(args.results_dir) / "baseline3_1_unk" / "predictions.jsonl"),
            ("baseline3_2_multitask", Path(args.results_dir) / "baseline3_2_multitask" / "predictions_cleaned.jsonl"),
            ("baseline3_3_semantic", Path(args.results_dir) / "baseline3_3_semantic" / "predictions.jsonl"),
            ("frontier_deepseek_v32_fewshot_cot", Path(args.results_dir) / "frontier_deepseek_v32_fewshot_cot" / "predictions.jsonl"),
            ("final", Path(args.results_dir) / "final" / "predictions.jsonl"),
            ("final_v2", Path(args.results_dir) / "final_v2" / "predictions.jsonl"),
            ("final_gap02_multitask_sigmoid", Path(args.results_dir) / "final_gap02_multitask_sigmoid" / "predictions.jsonl"),
            ("final_gap02_multitask_robustwpo", Path(args.results_dir) / "final_gap02_multitask_robustwpo" / "predictions.jsonl"),
            ("final_gap04_multitask_sigmoid", Path(args.results_dir) / "final_gap04_multitask_sigmoid" / "predictions.jsonl"),
            ("final_gap04_multitask_robustwpo", Path(args.results_dir) / "final_gap04_multitask_robustwpo" / "predictions.jsonl"),
            ("final_gap02_multitask_robustwpo_titleclean", Path(args.results_dir) / "final_gap02_multitask_robustwpo_titleclean" / "predictions.jsonl"),
            ("human_reference", Path(args.results_dir) / "human_reference" / "predictions.jsonl"),
        ]
    )

    reference_rows = []
    for name, path in prediction_files.items():
        if not path.exists():
            continue
        preds = [row["prediction"] for row in load_jsonl(path)]
        reference_rows.append({"method": name, **compute_reference_diagnostics(refs, preds)})

    train_lookup = build_train_lookup(Path(path) for path in args.train_data)
    dpo_rows = load_jsonl(Path(args.dpo_pairs))
    dpo_audit = audit_dpo_pairs(dpo_rows, train_lookup)

    if args.print_only:
        print(json.dumps({"reference_diagnostics": reference_rows, "dpo_audit": dpo_audit}, ensure_ascii=False, indent=2))
        return

    prefix = Path(args.output_prefix)
    write_json(prefix.with_name(prefix.name + "_reference.json"), {"rows": reference_rows})
    write_csv(prefix.with_name(prefix.name + "_reference.csv"), reference_rows)
    write_json(prefix.with_name(prefix.name + "_dpo.json"), dpo_audit)

    print(f"Wrote {prefix.with_name(prefix.name + '_reference.json')}")
    print(f"Wrote {prefix.with_name(prefix.name + '_reference.csv')}")
    print(f"Wrote {prefix.with_name(prefix.name + '_dpo.json')}")


if __name__ == "__main__":
    main()
