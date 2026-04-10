#!/usr/bin/env python3
"""Summarize reference-aware evaluation results by task-relevant subsets.

This script turns per-example reference-aware judge outputs into reviewer-facing
aggregates that are easier to cite in the paper. It focuses on two subset
families:
1. Whether the gold title contains common title-suffix characters.
2. Reference length bins, which roughly separate very short labels from longer
   title-like strings.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Iterable, List


DEFAULT_METHODS = [
    "baseline2",
    "baseline3_1_unk",
    "baseline3_2_multitask",
    "final_v2",
    "human_reference",
]

TITLE_SUFFIXES = set("經論記疏頌儀義傳錄贊序字品")
BAD_PATTERN = re.compile(
    r"[A-Za-z<>]|\[UNK\]|assistant|manuals|networks|shows|grat|dresses",
    re.IGNORECASE,
)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prediction_path(results_dir: Path, method: str) -> Path:
    if method == "baseline3_2_multitask":
        return results_dir / method / "predictions_cleaned.jsonl"
    return results_dir / method / "predictions.jsonl"


def judge_path(reference_eval_dir: Path, method: str) -> Path:
    return reference_eval_dir / f"{method}.json"


def subset_names(reference: str) -> list[str]:
    names = ["all"]

    if any(ch in TITLE_SUFFIXES for ch in reference):
        names.append("title_suffix")
    else:
        names.append("no_title_suffix")

    ref_len = len(reference)
    if ref_len <= 6:
        names.append("len_le_6")
    elif ref_len <= 9:
        names.append("len_7_9")
    else:
        names.append("len_ge_10")

    return names


def make_bucket() -> dict:
    return {
        "num_examples": 0,
        "exact_match": 0,
        "contamination_count": 0,
        "total_prediction_length": 0,
        "total_reference_length": 0,
        "sum_reference_agreement": 0,
        "sum_source_faithfulness": 0,
        "sum_title_style_fitness": 0,
        "sum_overall": 0,
    }


def finalize_bucket(method: str, subset: str, bucket: dict) -> OrderedDict:
    n = bucket["num_examples"] or 1
    avg_ref_len = bucket["total_reference_length"] / n
    avg_pred_len = bucket["total_prediction_length"] / n

    return OrderedDict(
        [
            ("method", method),
            ("subset", subset),
            ("num_examples", bucket["num_examples"]),
            ("exact_match_rate", round(bucket["exact_match"] / n, 4)),
            ("contamination_rate", round(bucket["contamination_count"] / n, 4)),
            ("length_ratio", round(avg_pred_len / avg_ref_len, 4) if avg_ref_len else None),
            ("mean_reference_agreement", round(bucket["sum_reference_agreement"] / n, 4)),
            ("mean_source_faithfulness", round(bucket["sum_source_faithfulness"] / n, 4)),
            ("mean_title_style_fitness", round(bucket["sum_title_style_fitness"] / n, 4)),
            ("mean_overall", round(bucket["sum_overall"] / n, 4)),
        ]
    )


def write_csv(path: Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze reference-aware evaluation by subset.")
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--reference-eval-dir", default="results/reference_eval_suite")
    parser.add_argument("--test-set", default="data/eval/test_set.jsonl")
    parser.add_argument("--output-dir", default="results/reference_eval_suite")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    reference_eval_dir = Path(args.reference_eval_dir)
    test_rows = load_jsonl(Path(args.test_set))

    subset_accumulators: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(make_bucket))
    all_scores_by_method: dict[str, list[dict]] = {}
    for method in args.methods:
        pred_rows = load_jsonl(prediction_path(results_dir, method))
        judge_rows = load_json(judge_path(reference_eval_dir, method))["scores"]
        if len(pred_rows) != len(test_rows) or len(judge_rows) != len(test_rows):
            raise ValueError(f"Length mismatch for {method}")

        all_scores_by_method[method] = judge_rows

        for idx, (test_row, pred_row, judge_row) in enumerate(zip(test_rows, pred_rows, judge_rows)):
            if pred_row["prediction"] != judge_row["prediction"]:
                raise ValueError(f"Prediction mismatch for {method} at index {idx}")
            if test_row["output"] != judge_row["reference"]:
                raise ValueError(f"Reference mismatch for {method} at index {idx}")

            reference = test_row["output"]
            prediction = pred_row["prediction"]
            for subset in subset_names(reference):
                bucket = subset_accumulators[method][subset]
                bucket["num_examples"] += 1
                bucket["exact_match"] += int(prediction == reference)
                bucket["contamination_count"] += int(bool(BAD_PATTERN.search(prediction)))
                bucket["total_prediction_length"] += len(prediction)
                bucket["total_reference_length"] += len(reference)
                bucket["sum_reference_agreement"] += int(judge_row["reference_agreement"])
                bucket["sum_source_faithfulness"] += int(judge_row["source_faithfulness"])
                bucket["sum_title_style_fitness"] += int(judge_row["title_style_fitness"])
                bucket["sum_overall"] += int(judge_row["overall"])

    summary_rows = []
    subset_order = ["all", "title_suffix", "no_title_suffix", "len_le_6", "len_7_9", "len_ge_10"]
    for method in args.methods:
        for subset in subset_order:
            if subset not in subset_accumulators[method]:
                continue
            summary_rows.append(finalize_bucket(method, subset, subset_accumulators[method][subset]))

    pairings = [
        ("baseline3_2_multitask", "baseline3_1_unk"),
        ("baseline3_2_multitask", "final_v2"),
        ("baseline3_1_unk", "final_v2"),
    ]
    pairwise_rows = []
    for method_a, method_b in pairings:
        if method_a not in all_scores_by_method or method_b not in all_scores_by_method:
            continue

        counts = defaultdict(lambda: {"a_wins": 0, "ties": 0, "b_wins": 0})
        for test_row, score_a, score_b in zip(test_rows, all_scores_by_method[method_a], all_scores_by_method[method_b]):
            subsets = subset_names(test_row["output"])
            for subset in subsets:
                a_val = int(score_a["overall"])
                b_val = int(score_b["overall"])
                if a_val > b_val:
                    counts[subset]["a_wins"] += 1
                elif b_val > a_val:
                    counts[subset]["b_wins"] += 1
                else:
                    counts[subset]["ties"] += 1

        for subset in subset_order:
            if subset not in counts:
                continue
            pairwise_rows.append(
                OrderedDict(
                    [
                        ("method_a", method_a),
                        ("method_b", method_b),
                        ("subset", subset),
                        ("a_wins", counts[subset]["a_wins"]),
                        ("ties", counts[subset]["ties"]),
                        ("b_wins", counts[subset]["b_wins"]),
                    ]
                )
            )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_json = output_dir / "subset_summary.json"
    summary_csv = output_dir / "subset_summary.csv"
    pairwise_json = output_dir / "pairwise_overall_wins.json"
    pairwise_csv = output_dir / "pairwise_overall_wins.csv"

    summary_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pairwise_json.write_text(json.dumps(pairwise_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(summary_csv, summary_rows)
    write_csv(pairwise_csv, pairwise_rows)

    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {pairwise_json}")
    print(f"Wrote {pairwise_csv}")


if __name__ == "__main__":
    main()
