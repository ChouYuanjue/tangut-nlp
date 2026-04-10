#!/usr/bin/env python3
"""Audit exact and character n-gram overlap across Tangut data splits."""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path


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


def final_target(row: dict) -> str:
    output = row["output"]
    if "</literal> " in output:
        return output.split("</literal> ", 1)[1]
    return output


def char_ngrams(text: str, n: int) -> set[str]:
    if len(text) < n:
        return set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def max_jaccard(target: str, pool: list[str], n: int) -> float:
    grams = char_ngrams(target, n)
    if not grams:
        return 0.0
    best = 0.0
    for candidate in pool:
        cand_grams = char_ngrams(candidate, n)
        if not cand_grams:
            continue
        union = grams | cand_grams
        if not union:
            continue
        score = len(grams & cand_grams) / len(union)
        if score > best:
            best = score
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Leakage audit for Tangut data splits.")
    parser.add_argument("--train-set", default="data/sft/babelstone_sft.jsonl")
    parser.add_argument("--dev-set", default="data/eval/dev_set.jsonl")
    parser.add_argument("--test-set", default="data/eval/test_set.jsonl")
    parser.add_argument("--synthetic-set", default="data/sft/synthetic_sft_multitask.jsonl")
    parser.add_argument("--fewshot-ids", nargs="*", type=int, default=[1, 6, 15, 16])
    parser.add_argument("--output-dir", default="results/analysis/leakage_audit")
    args = parser.parse_args()

    train_rows = load_jsonl(Path(args.train_set))
    dev_rows = load_jsonl(Path(args.dev_set))
    test_rows = load_jsonl(Path(args.test_set))
    synth_rows = load_jsonl(Path(args.synthetic_set))

    train_inputs = {row["input"] for row in train_rows}
    dev_inputs = {row["input"] for row in dev_rows}
    synth_inputs = {row["input"] for row in synth_rows}

    train_pairs = {(row["input"], row["output"]) for row in train_rows}
    dev_pairs = {(row["input"], row["output"]) for row in dev_rows}
    synth_pairs = {(row["input"], final_target(row)) for row in synth_rows}

    train_targets = [row["output"] for row in train_rows]
    dev_targets = [row["output"] for row in dev_rows]
    synth_targets = [final_target(row) for row in synth_rows]
    fewshot_targets = [dev_rows[idx]["output"] for idx in args.fewshot_ids]

    summary = OrderedDict(
        [
            (
                "exact_overlap",
                OrderedDict(
                    [
                        ("test_pair_in_train", sum((row["input"], row["output"]) in train_pairs for row in test_rows)),
                        ("test_pair_in_dev", sum((row["input"], row["output"]) in dev_pairs for row in test_rows)),
                        (
                            "test_pair_in_synthetic_final_targets",
                            sum((row["input"], row["output"]) in synth_pairs for row in test_rows),
                        ),
                        ("test_input_in_train", sum(row["input"] in train_inputs for row in test_rows)),
                        ("test_input_in_dev", sum(row["input"] in dev_inputs for row in test_rows)),
                        ("test_input_in_synthetic", sum(row["input"] in synth_inputs for row in test_rows)),
                        ("test_target_in_train", sum(row["output"] in set(train_targets) for row in test_rows)),
                        ("test_target_in_dev", sum(row["output"] in set(dev_targets) for row in test_rows)),
                        ("test_target_in_synthetic", sum(row["output"] in set(synth_targets) for row in test_rows)),
                        (
                            "frontier_fewshot_target_in_test",
                            sum(target in {row["output"] for row in test_rows} for target in fewshot_targets),
                        ),
                    ]
                ),
            )
        ]
    )

    per_item_rows = []
    for idx, row in enumerate(test_rows):
        target = row["output"]
        per_item_rows.append(
            OrderedDict(
                [
                    ("index", idx),
                    ("input", row["input"]),
                    ("reference", target),
                    ("max_jaccard_train_2gram", round(max_jaccard(target, train_targets, 2), 4)),
                    ("max_jaccard_dev_2gram", round(max_jaccard(target, dev_targets, 2), 4)),
                    ("max_jaccard_synth_2gram", round(max_jaccard(target, synth_targets, 2), 4)),
                    ("max_jaccard_train_3gram", round(max_jaccard(target, train_targets, 3), 4)),
                    ("max_jaccard_dev_3gram", round(max_jaccard(target, dev_targets, 3), 4)),
                    ("max_jaccard_synth_3gram", round(max_jaccard(target, synth_targets, 3), 4)),
                    ("max_jaccard_train_4gram", round(max_jaccard(target, train_targets, 4), 4)),
                    ("max_jaccard_dev_4gram", round(max_jaccard(target, dev_targets, 4), 4)),
                    ("max_jaccard_synth_4gram", round(max_jaccard(target, synth_targets, 4), 4)),
                ]
            )
        )

    aggregate_rows = []
    for prefix in ["train", "dev", "synth"]:
        aggregate_rows.append(
            OrderedDict(
                [
                    ("pool", prefix),
                    (
                        "mean_max_jaccard_2gram",
                        round(sum(row[f"max_jaccard_{prefix}_2gram"] for row in per_item_rows) / len(per_item_rows), 4),
                    ),
                    (
                        "max_max_jaccard_2gram",
                        round(max(row[f"max_jaccard_{prefix}_2gram"] for row in per_item_rows), 4),
                    ),
                    (
                        "mean_max_jaccard_3gram",
                        round(sum(row[f"max_jaccard_{prefix}_3gram"] for row in per_item_rows) / len(per_item_rows), 4),
                    ),
                    (
                        "max_max_jaccard_3gram",
                        round(max(row[f"max_jaccard_{prefix}_3gram"] for row in per_item_rows), 4),
                    ),
                    (
                        "mean_max_jaccard_4gram",
                        round(sum(row[f"max_jaccard_{prefix}_4gram"] for row in per_item_rows) / len(per_item_rows), 4),
                    ),
                    (
                        "max_max_jaccard_4gram",
                        round(max(row[f"max_jaccard_{prefix}_4gram"] for row in per_item_rows), 4),
                    ),
                ]
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    per_item_path = output_dir / "per_item.csv"
    aggregate_path = output_dir / "ngram_overlap_summary.csv"

    payload = OrderedDict([("summary", summary), ("ngram_summary", aggregate_rows)])
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(per_item_path, per_item_rows)
    write_csv(aggregate_path, aggregate_rows)

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {summary_path}")
    print(f"Wrote {per_item_path}")
    print(f"Wrote {aggregate_path}")


if __name__ == "__main__":
    main()
