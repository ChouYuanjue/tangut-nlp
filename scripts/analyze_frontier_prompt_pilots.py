#!/usr/bin/env python3
"""Summarize stored frontier prompt-sensitivity pilot runs.

This script does not call the frontier API. It compares existing pilot outputs
saved in the repository for different prompt variants and reports exact match
and chrF++ on the shared subset of examples.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import sacrebleu


DEFAULT_VARIANTS = OrderedDict(
    [
        ("strict_json_title_cot", "results/frontier_deepseek_v32_dict_cot/predictions_pilot.jsonl"),
        ("strict_title_only_cot", "results/frontier_deepseek_v32_dict_cot/predictions_pilot_v2.jsonl"),
        ("fewshot_title_only_cot", "results/frontier_deepseek_v32_dict_cot/predictions_pilot_fewshot.jsonl"),
    ]
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def corpus_chrf(rows: list[dict]) -> float:
    hyps = [row["prediction"] for row in rows]
    refs = [row["reference"] for row in rows]
    return sacrebleu.corpus_chrf(hyps, [refs], char_order=6, word_order=2, beta=2).score


def summarize_rows(name: str, rows: list[dict]) -> OrderedDict:
    exact = sum(int(row["prediction"] == row["reference"]) for row in rows)
    return OrderedDict(
        [
            ("variant", name),
            ("num_examples", len(rows)),
            ("exact_match", exact),
            ("exact_match_rate", round(exact / len(rows), 4) if rows else 0.0),
            ("corpus_chrf", round(corpus_chrf(rows), 4) if rows else 0.0),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze stored frontier prompt-sensitivity pilot runs.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--output",
        default="results/analysis/frontier_prompt_sensitivity/summary.json",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    rows_by_variant = {
        name: load_jsonl(repo_root / rel_path) for name, rel_path in DEFAULT_VARIANTS.items()
    }

    full_rows = [summarize_rows(name, rows) for name, rows in rows_by_variant.items()]

    shared_inputs = sorted(
        set.intersection(*[{row["input"] for row in rows} for rows in rows_by_variant.values() if rows])
    )
    shared_rows = []
    for name, rows in rows_by_variant.items():
        row_map = {row["input"]: row for row in rows}
        subset = [row_map[input_text] for input_text in shared_inputs]
        summary = summarize_rows(name, subset)
        summary["subset"] = "shared_all_variants"
        shared_rows.append(summary)

    pairwise_rows = []
    pairs = [
        ("strict_title_only_cot", "fewshot_title_only_cot"),
        ("strict_json_title_cot", "fewshot_title_only_cot"),
    ]
    for left, right in pairs:
        left_map = {row["input"]: row for row in rows_by_variant[left]}
        right_map = {row["input"]: row for row in rows_by_variant[right]}
        shared = sorted(set(left_map) & set(right_map))
        left_subset = [left_map[input_text] for input_text in shared]
        right_subset = [right_map[input_text] for input_text in shared]
        pairwise_rows.append(
            OrderedDict(
                [
                    ("left_variant", left),
                    ("right_variant", right),
                    ("shared_examples", len(shared)),
                    ("left_exact", sum(int(row["prediction"] == row["reference"]) for row in left_subset)),
                    ("right_exact", sum(int(row["prediction"] == row["reference"]) for row in right_subset)),
                    ("left_chrf", round(corpus_chrf(left_subset), 4)),
                    ("right_chrf", round(corpus_chrf(right_subset), 4)),
                ]
            )
        )

    payload = OrderedDict(
        [
            ("metadata", {"temperature": 0.0, "sampling_seed_applicable": False}),
            ("full_variant_summary", full_rows),
            ("shared_subset_summary", shared_rows),
            ("pairwise_shared_subset_summary", pairwise_rows),
        ]
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_path.with_name("full_variant_summary.csv"), full_rows)
    write_csv(output_path.with_name("shared_subset_summary.csv"), shared_rows)
    write_csv(output_path.with_name("pairwise_shared_subset_summary.csv"), pairwise_rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
