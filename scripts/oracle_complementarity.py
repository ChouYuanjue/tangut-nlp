#!/usr/bin/env python3
"""Analyze complementarity between two Tangut translation systems.

This script focuses on reviewer-facing questions:
1. How much do two systems overlap on exact matches?
2. How large is the oracle upper bound if we could pick the better output?
3. Do agreement/disagreement regions behave like a useful uncertainty signal?
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import OrderedDict
from pathlib import Path

import sacrebleu


TITLE_SUFFIXES = set("經論記疏頌儀義傳錄贊序字品")


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prediction_path(results_dir: Path, method: str) -> Path:
    if method == "baseline3_2_multitask":
        return results_dir / method / "predictions_cleaned.jsonl"
    return results_dir / method / "predictions.jsonl"


def judge_path(results_dir: Path, reference_eval_dir: Path, method: str) -> Path:
    candidates = [
        reference_eval_dir / f"{method}.json",
        results_dir / method / "reference_aware_judge.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing judge output for {method}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def sent_chrf(hypothesis: str, reference: str) -> float:
    return sacrebleu.corpus_chrf(
        [hypothesis],
        [[reference]],
        char_order=6,
        word_order=2,
        beta=2,
    ).score


def subset_name(reference: str, same_prediction: bool) -> str:
    if same_prediction:
        return "same_prediction"
    if any(ch in TITLE_SUFFIXES for ch in reference):
        return "disagree_title_suffix"
    return "disagree_no_title_suffix"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze oracle complementarity between two methods.")
    parser.add_argument("--method-a", default="frontier_deepseek_v32_fewshot_cot")
    parser.add_argument("--method-b", default="final_gap04_multitask_sigmoid")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--reference-eval-dir", default="results/reference_eval_suite_combined")
    parser.add_argument("--test-set", default="data/eval/test_set.jsonl")
    parser.add_argument("--output-dir", default="results/analysis/oracle_complementarity")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    reference_eval_dir = Path(args.reference_eval_dir)
    test_rows = load_jsonl(Path(args.test_set))

    pred_a = load_jsonl(prediction_path(results_dir, args.method_a))
    pred_b = load_jsonl(prediction_path(results_dir, args.method_b))
    judge_a = load_json(judge_path(results_dir, reference_eval_dir, args.method_a))["scores"]
    judge_b = load_json(judge_path(results_dir, reference_eval_dir, args.method_b))["scores"]

    if not (len(test_rows) == len(pred_a) == len(pred_b) == len(judge_a) == len(judge_b)):
        raise ValueError("Mismatched lengths across predictions, references, and judges")

    refs = [row["output"] for row in test_rows]
    hyps_a = [row["prediction"] for row in pred_a]
    hyps_b = [row["prediction"] for row in pred_b]

    exact_a = [hyp == ref for hyp, ref in zip(hyps_a, refs)]
    exact_b = [hyp == ref for hyp, ref in zip(hyps_b, refs)]
    same_prediction = [a == b for a, b in zip(hyps_a, hyps_b)]

    per_item_rows = []
    oracle_chrf_hyps = []
    oracle_judge_rows = []
    oracle_exact = 0
    judge_wins_a = 0
    judge_wins_b = 0
    judge_ties = 0
    chrf_wins_a = 0
    chrf_wins_b = 0
    chrf_ties = 0

    for idx, (test_row, row_a, row_b, row_judge_a, row_judge_b) in enumerate(
        zip(test_rows, pred_a, pred_b, judge_a, judge_b)
    ):
        ref = test_row["output"]
        hyp_a = row_a["prediction"]
        hyp_b = row_b["prediction"]

        chrf_a = sent_chrf(hyp_a, ref)
        chrf_b = sent_chrf(hyp_b, ref)
        if chrf_a > chrf_b:
            chrf_wins_a += 1
            oracle_chrf_hyps.append(hyp_a)
            chrf_winner = args.method_a
        elif chrf_b > chrf_a:
            chrf_wins_b += 1
            oracle_chrf_hyps.append(hyp_b)
            chrf_winner = args.method_b
        else:
            chrf_ties += 1
            oracle_chrf_hyps.append(hyp_a)
            chrf_winner = "tie"

        overall_a = int(row_judge_a["overall"])
        overall_b = int(row_judge_b["overall"])
        if overall_a > overall_b:
            judge_wins_a += 1
            oracle_judge_rows.append(row_judge_a)
            judge_winner = args.method_a
        elif overall_b > overall_a:
            judge_wins_b += 1
            oracle_judge_rows.append(row_judge_b)
            judge_winner = args.method_b
        else:
            judge_ties += 1
            oracle_judge_rows.append(row_judge_a)
            judge_winner = "tie"

        oracle_exact += int(exact_a[idx] or exact_b[idx])

        per_item_rows.append(
            OrderedDict(
                [
                    ("index", idx),
                    ("subset", subset_name(ref, same_prediction[idx])),
                    ("input", test_row["input"]),
                    ("reference", ref),
                    ("prediction_a", hyp_a),
                    ("prediction_b", hyp_b),
                    ("same_prediction", same_prediction[idx]),
                    ("exact_a", exact_a[idx]),
                    ("exact_b", exact_b[idx]),
                    ("sentence_chrf_a", round(chrf_a, 4)),
                    ("sentence_chrf_b", round(chrf_b, 4)),
                    ("chrf_winner", chrf_winner),
                    ("judge_overall_a", overall_a),
                    ("judge_overall_b", overall_b),
                    ("judge_winner", judge_winner),
                ]
            )
        )

    corpus_chrf_a = sacrebleu.corpus_chrf(hyps_a, [refs], char_order=6, word_order=2, beta=2).score
    corpus_chrf_b = sacrebleu.corpus_chrf(hyps_b, [refs], char_order=6, word_order=2, beta=2).score
    corpus_chrf_oracle = sacrebleu.corpus_chrf(
        oracle_chrf_hyps, [refs], char_order=6, word_order=2, beta=2
    ).score

    summary = OrderedDict(
        [
            ("method_a", args.method_a),
            ("method_b", args.method_b),
            ("num_examples", len(test_rows)),
            (
                "exact_match",
                OrderedDict(
                    [
                        ("method_a", sum(exact_a)),
                        ("method_b", sum(exact_b)),
                        ("union", oracle_exact),
                        ("only_method_a", sum(a and not b for a, b in zip(exact_a, exact_b))),
                        ("only_method_b", sum(b and not a for a, b in zip(exact_a, exact_b))),
                        ("both", sum(a and b for a, b in zip(exact_a, exact_b))),
                    ]
                ),
            ),
            (
                "prediction_agreement",
                OrderedDict(
                    [
                        ("same_prediction_count", sum(same_prediction)),
                        ("same_prediction_rate", round(sum(same_prediction) / len(test_rows), 4)),
                        ("different_prediction_count", len(test_rows) - sum(same_prediction)),
                        (
                            "different_prediction_rate",
                            round((len(test_rows) - sum(same_prediction)) / len(test_rows), 4),
                        ),
                    ]
                ),
            ),
            (
                "sentence_chrf",
                OrderedDict(
                    [
                        ("method_a_mean", round(sum(row["sentence_chrf_a"] for row in per_item_rows) / len(test_rows), 4)),
                        ("method_b_mean", round(sum(row["sentence_chrf_b"] for row in per_item_rows) / len(test_rows), 4)),
                        (
                            "oracle_mean",
                            round(
                                sum(max(row["sentence_chrf_a"], row["sentence_chrf_b"]) for row in per_item_rows)
                                / len(test_rows),
                                4,
                            ),
                        ),
                        ("method_a_better_items", chrf_wins_a),
                        ("method_b_better_items", chrf_wins_b),
                        ("ties", chrf_ties),
                    ]
                ),
            ),
            (
                "corpus_chrf",
                OrderedDict(
                    [
                        ("method_a", round(corpus_chrf_a, 4)),
                        ("method_b", round(corpus_chrf_b, 4)),
                        ("oracle", round(corpus_chrf_oracle, 4)),
                    ]
                ),
            ),
            (
                "judge_overall",
                OrderedDict(
                    [
                        (
                            "method_a_mean",
                            round(sum(int(row["overall"]) for row in judge_a) / len(judge_a), 4),
                        ),
                        (
                            "method_b_mean",
                            round(sum(int(row["overall"]) for row in judge_b) / len(judge_b), 4),
                        ),
                        (
                            "oracle_mean",
                            round(sum(int(row["overall"]) for row in oracle_judge_rows) / len(oracle_judge_rows), 4),
                        ),
                        ("method_a_wins", judge_wins_a),
                        ("method_b_wins", judge_wins_b),
                        ("ties", judge_ties),
                    ]
                ),
            ),
            (
                "agreement_regions",
                OrderedDict(
                    [
                        (
                            "same_prediction",
                            OrderedDict(
                                [
                                    ("num_examples", sum(row["same_prediction"] for row in per_item_rows)),
                                    (
                                        "any_exact_rate",
                                        round(
                                            sum(
                                                row["same_prediction"] and (row["exact_a"] or row["exact_b"])
                                                for row in per_item_rows
                                            )
                                            / max(1, sum(row["same_prediction"] for row in per_item_rows)),
                                            4,
                                        ),
                                    ),
                                ]
                            ),
                        ),
                        (
                            "different_prediction",
                            OrderedDict(
                                [
                                    ("num_examples", sum(not row["same_prediction"] for row in per_item_rows)),
                                    (
                                        "any_exact_rate",
                                        round(
                                            sum(
                                                (not row["same_prediction"]) and (row["exact_a"] or row["exact_b"])
                                                for row in per_item_rows
                                            )
                                            / max(1, sum(not row["same_prediction"] for row in per_item_rows)),
                                            4,
                                        ),
                                    ),
                                    (
                                        "mean_best_judge_overall",
                                        round(
                                            sum(
                                                (not row["same_prediction"])
                                                * max(row["judge_overall_a"], row["judge_overall_b"])
                                                for row in per_item_rows
                                            )
                                            / max(1, sum(not row["same_prediction"] for row in per_item_rows)),
                                            4,
                                        ),
                                    ),
                                ]
                            ),
                        ),
                    ]
                ),
            ),
        ]
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / f"{args.method_a}__vs__{args.method_b}.json"
    per_item_path = output_dir / f"{args.method_a}__vs__{args.method_b}.csv"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(per_item_path, per_item_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote {summary_path}")
    print(f"Wrote {per_item_path}")


if __name__ == "__main__":
    main()
