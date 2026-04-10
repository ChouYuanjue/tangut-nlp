#!/usr/bin/env python3
"""Run the reference-aware judge over a curated set of Tangut-NLP systems."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import OrderedDict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from eval.reference_aware_judge import ReferenceAwareJudge


DEFAULT_METHODS = [
    "baseline2",
    "baseline3_1_unk",
    "baseline3_2_multitask",
    "final_v2",
    "human_reference",
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prediction_path(results_dir: Path, method: str) -> Path:
    custom = {
        "baseline3_2_multitask": results_dir / method / "predictions_cleaned.jsonl",
    }
    return custom.get(method, results_dir / method / "predictions.jsonl")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reference-aware evaluation suite.")
    parser.add_argument(
        "--methods",
        nargs="*",
        default=DEFAULT_METHODS,
        help="Methods to evaluate.",
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--test-set", default="data/eval/test_set.jsonl")
    parser.add_argument(
        "--output-dir",
        default="results/reference_eval_suite",
        help="Directory to store per-method and summary outputs.",
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--api-key", default=None, help="Optional explicit Azure API key override.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    test_set = load_jsonl(Path(args.test_set))
    judge = ReferenceAwareJudge(api_key=args.api_key, mock=args.mock, timeout=args.timeout)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for method in args.methods:
        pred_path = prediction_path(results_dir, method)
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions for {method}: {pred_path}")

        predictions = load_jsonl(pred_path)
        if len(predictions) != len(test_set):
            raise ValueError(
                f"Prediction/reference length mismatch for {method}: "
                f"{len(predictions)} vs {len(test_set)}"
            )

        items = []
        for pred, ref in zip(predictions, test_set):
            items.append(
                {
                    "input": pred["input"],
                    "reference": ref["output"],
                    "prediction": pred["prediction"],
                    "method": method,
                }
            )

        print(f"\n=== Running reference-aware judge: {method} ===")
        result = judge.score_batch(items)

        out_path = output_dir / f"{method}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        summary_rows.append(
            OrderedDict(
                [
                    ("method", method),
                    ("num_examples", len(result["scores"])),
                    ("mean_reference_agreement", round(result["mean_reference_agreement"], 4)),
                    ("mean_source_faithfulness", round(result["mean_source_faithfulness"], 4)),
                    ("mean_title_style_fitness", round(result["mean_title_style_fitness"], 4)),
                    ("mean_overall", round(result["mean_overall"], 4)),
                    ("output_json", str(out_path)),
                ]
            )
        )

    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    summary_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(summary_csv, summary_rows)

    print("\n=== Reference-aware suite summary ===")
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
