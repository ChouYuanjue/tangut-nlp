#!/usr/bin/env python3
"""Evaluate a real oracle portability probe without external judge calls."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.chrf_scorer import ChrFScorer
from eval.lexical_coverage import LexicalCoverageScorer


BAD_PATTERN = re.compile(
    r"[A-Za-z<>]|\[UNK\]|assistant|步骤|解释|分析|reason|translation",
    re.IGNORECASE,
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate oracle portability predictions with exact match, chrF++, and diagnostic metrics."
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--test-set", required=True)
    parser.add_argument("--reward-dict", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--details-output",
        default=None,
        help="Optional JSONL path for per-example diagnostics.",
    )
    args = parser.parse_args()

    pred_rows = load_jsonl(Path(args.predictions))
    test_rows = load_jsonl(Path(args.test_set))
    if len(pred_rows) != len(test_rows):
        raise ValueError(
            f"Prediction/reference length mismatch: {len(pred_rows)} vs {len(test_rows)}"
        )

    predictions = [row.get("prediction", "").strip() for row in pred_rows]
    references = [row.get("output", "").strip() for row in test_rows]
    inputs = [row.get("input", "") for row in pred_rows]

    exact = sum(int(pred == ref) for pred, ref in zip(predictions, references))
    contamination = sum(int(bool(BAD_PATTERN.search(pred))) for pred in predictions)

    chrf = ChrFScorer().score(predictions, references)
    lex = LexicalCoverageScorer(args.reward_dict).score_batch(list(zip(inputs, predictions)))

    metrics = {
        "num_examples": len(pred_rows),
        "exact_match": exact,
        "exact_match_rate": round(exact / len(pred_rows), 4) if pred_rows else 0.0,
        "contamination_count": contamination,
        "contamination_rate": round(contamination / len(pred_rows), 4) if pred_rows else 0.0,
        "avg_prediction_length": round(
            sum(len(pred) for pred in predictions) / len(predictions), 4
        )
        if predictions
        else 0.0,
        "avg_reference_length": round(
            sum(len(ref) for ref in references) / len(references), 4
        )
        if references
        else 0.0,
        "chrf": {
            "corpus_chrf": round(chrf["corpus_chrf"], 4),
            "mean_sentence_chrf": round(chrf["mean_sentence_chrf"], 4),
        },
        "lexical_coverage": {
            "mean": round(lex["mean"], 4),
            "min": round(lex["min"], 4),
            "max": round(lex["max"], 4),
        },
    }

    dump_json(Path(args.output), metrics)

    if args.details_output:
        details_path = Path(args.details_output)
        details_path.parent.mkdir(parents=True, exist_ok=True)
        with details_path.open("w", encoding="utf-8") as f:
            for idx, (pred_row, test_row, sent_chrf) in enumerate(
                zip(pred_rows, test_rows, chrf["per_sentence"]),
                start=1,
            ):
                payload = {
                    "index": idx,
                    "input": pred_row.get("input", ""),
                    "input_ids": test_row.get("input_ids"),
                    "prediction": pred_row.get("prediction", "").strip(),
                    "reference": test_row.get("output", "").strip(),
                    "exact": int(
                        pred_row.get("prediction", "").strip() == test_row.get("output", "").strip()
                    ),
                    "sentence_chrf": round(sent_chrf, 4),
                    "contaminated": int(bool(BAD_PATTERN.search(pred_row.get("prediction", "")))),
                    "metadata": test_row.get("metadata", {}),
                }
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
