#!/usr/bin/env python3
"""Bootstrap confidence intervals and paired comparisons for Tangut systems."""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import OrderedDict
from pathlib import Path

import sacrebleu


DEFAULT_METHODS = [
    "frontier_deepseek_v32_fewshot_cot",
    "hybrid_select_frontier_local_gpt54",
    "baseline3_2_multitask",
    "final_gap04_multitask_sigmoid",
]

DEFAULT_COMPARISONS = [
    ("frontier_deepseek_v32_fewshot_cot", "hybrid_select_frontier_local_gpt54"),
    ("final_gap04_multitask_sigmoid", "hybrid_select_frontier_local_gpt54"),
    ("frontier_deepseek_v32_fewshot_cot", "final_gap04_multitask_sigmoid"),
    ("baseline3_2_multitask", "final_gap04_multitask_sigmoid"),
    ("baseline3_2_multitask", "final_v2"),
]


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


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def corpus_chrf(hyps: list[str], refs: list[str]) -> float:
    return sacrebleu.corpus_chrf(hyps, [refs], char_order=6, word_order=2, beta=2).score


def bootstrap_sample_indices(n: int, rng: random.Random) -> list[int]:
    return [rng.randrange(n) for _ in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap significance for Tangut results.")
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--reference-eval-dir", default="results/reference_eval_suite_combined")
    parser.add_argument("--test-set", default="data/eval/test_set.jsonl")
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results/analysis/bootstrap_significance")
    args = parser.parse_args()

    comparisons = [
        pair
        for pair in DEFAULT_COMPARISONS
        if pair[0] in args.methods and pair[1] in args.methods + ["final_v2"]
    ]

    results_dir = Path(args.results_dir)
    reference_eval_dir = Path(args.reference_eval_dir)
    test_rows = load_jsonl(Path(args.test_set))
    refs = [row["output"] for row in test_rows]
    n = len(test_rows)

    needed_methods = set(args.methods) | {item for pair in comparisons for item in pair}

    method_payloads = {}
    for method in needed_methods:
        pred_rows = load_jsonl(prediction_path(results_dir, method))
        judge_rows = load_json(judge_path(results_dir, reference_eval_dir, method))["scores"]
        method_payloads[method] = {
            "hyps": [row["prediction"] for row in pred_rows],
            "judge_overall": [int(row["overall"]) for row in judge_rows],
            "exact": [int(pred_rows[i]["prediction"] == refs[i]) for i in range(n)],
        }

    rng = random.Random(args.seed)
    bootstrap_indices = [bootstrap_sample_indices(n, rng) for _ in range(args.num_samples)]
    point_rows = []
    bootstrap_store: dict[str, dict[str, list[float]]] = {}
    for method in needed_methods:
        payload = method_payloads[method]
        hyps = payload["hyps"]
        judge = payload["judge_overall"]
        exact = payload["exact"]
        chrf_samples = []
        judge_samples = []
        exact_samples = []
        for sample_idx in bootstrap_indices:
            sample_hyps = [hyps[i] for i in sample_idx]
            sample_refs = [refs[i] for i in sample_idx]
            chrf_samples.append(corpus_chrf(sample_hyps, sample_refs))
            judge_samples.append(sum(judge[i] for i in sample_idx) / n)
            exact_samples.append(sum(exact[i] for i in sample_idx) / n)
        bootstrap_store[method] = {
            "chrf": chrf_samples,
            "judge_overall": judge_samples,
            "exact_rate": exact_samples,
        }
        point_rows.append(
            OrderedDict(
                [
                    ("method", method),
                    ("corpus_chrf", round(corpus_chrf(hyps, refs), 4)),
                    ("corpus_chrf_ci_low", round(quantile(chrf_samples, 0.025), 4)),
                    ("corpus_chrf_ci_high", round(quantile(chrf_samples, 0.975), 4)),
                    ("mean_judge_overall", round(sum(judge) / n, 4)),
                    ("mean_judge_overall_ci_low", round(quantile(judge_samples, 0.025), 4)),
                    ("mean_judge_overall_ci_high", round(quantile(judge_samples, 0.975), 4)),
                    ("exact_match_rate", round(sum(exact) / n, 4)),
                    ("exact_match_rate_ci_low", round(quantile(exact_samples, 0.025), 4)),
                    ("exact_match_rate_ci_high", round(quantile(exact_samples, 0.975), 4)),
                ]
            )
        )

    comparison_rows = []
    for method_a, method_b in comparisons:
        chrf_delta = [
            b - a for a, b in zip(bootstrap_store[method_a]["chrf"], bootstrap_store[method_b]["chrf"])
        ]
        judge_delta = [
            b - a
            for a, b in zip(
                bootstrap_store[method_a]["judge_overall"],
                bootstrap_store[method_b]["judge_overall"],
            )
        ]
        exact_delta = [
            b - a
            for a, b in zip(
                bootstrap_store[method_a]["exact_rate"],
                bootstrap_store[method_b]["exact_rate"],
            )
        ]

        comparison_rows.append(
            OrderedDict(
                [
                    ("method_a", method_a),
                    ("method_b", method_b),
                    (
                        "delta_corpus_chrf_b_minus_a",
                        round(corpus_chrf(method_payloads[method_b]["hyps"], refs) - corpus_chrf(method_payloads[method_a]["hyps"], refs), 4),
                    ),
                    ("delta_corpus_chrf_ci_low", round(quantile(chrf_delta, 0.025), 4)),
                    ("delta_corpus_chrf_ci_high", round(quantile(chrf_delta, 0.975), 4)),
                    (
                        "delta_mean_judge_overall_b_minus_a",
                        round(
                            (sum(method_payloads[method_b]["judge_overall"]) - sum(method_payloads[method_a]["judge_overall"])) / n,
                            4,
                        ),
                    ),
                    ("delta_mean_judge_ci_low", round(quantile(judge_delta, 0.025), 4)),
                    ("delta_mean_judge_ci_high", round(quantile(judge_delta, 0.975), 4)),
                    (
                        "delta_exact_rate_b_minus_a",
                        round(
                            (sum(method_payloads[method_b]["exact"]) - sum(method_payloads[method_a]["exact"])) / n,
                            4,
                        ),
                    ),
                    ("delta_exact_ci_low", round(quantile(exact_delta, 0.025), 4)),
                    ("delta_exact_ci_high", round(quantile(exact_delta, 0.975), 4)),
                ]
            )
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    points_json = output_dir / "method_confidence_intervals.json"
    points_csv = output_dir / "method_confidence_intervals.csv"
    comps_json = output_dir / "paired_comparisons.json"
    comps_csv = output_dir / "paired_comparisons.csv"

    points_json.write_text(json.dumps(point_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    comps_json.write_text(json.dumps(comparison_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(points_csv, point_rows)
    write_csv(comps_csv, comparison_rows)

    print(json.dumps({"methods": point_rows, "comparisons": comparison_rows}, ensure_ascii=False, indent=2))
    print(f"Wrote {points_json}")
    print(f"Wrote {points_csv}")
    print(f"Wrote {comps_json}")
    print(f"Wrote {comps_csv}")


if __name__ == "__main__":
    main()
