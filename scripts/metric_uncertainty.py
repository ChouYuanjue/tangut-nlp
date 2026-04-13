#!/usr/bin/env python3
"""Compute bootstrap chrF++ intervals and Wilson exact-match intervals.

This is a lightweight reviewer-facing utility for reporting uncertainty on
stored prediction files. Each prediction row must contain ``prediction`` and
``reference`` fields.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import OrderedDict
from pathlib import Path

import sacrebleu


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


def quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def corpus_chrf(hyps: list[str], refs: list[str]) -> float:
    return sacrebleu.corpus_chrf(hyps, [refs], char_order=6, word_order=2, beta=2).score


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    p = successes / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denom
    margin = (
        z
        * math.sqrt((p * (1 - p) / total) + (z**2) / (4 * total * total))
        / denom
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def bootstrap_indices(n: int, rng: random.Random) -> list[int]:
    return [rng.randrange(n) for _ in range(n)]


def parse_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected NAME=PATH, got: {spec}")
    name, raw_path = spec.split("=", 1)
    return name.strip(), Path(raw_path).expanduser()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute bootstrap chrF++ CIs and Wilson exact-match intervals."
    )
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        help="Repeated NAME=PATH specification for prediction files.",
    )
    parser.add_argument("--num-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = []
    for spec in args.pred:
        name, path = parse_spec(spec)
        pred_rows = load_jsonl(path)
        refs = [row["reference"] for row in pred_rows]
        hyps = [row["prediction"] for row in pred_rows]
        n = len(pred_rows)
        exact = sum(int(h == r) for h, r in zip(hyps, refs))

        chrf_samples = []
        for _ in range(args.num_samples):
            sample_idx = bootstrap_indices(n, rng)
            sample_hyps = [hyps[i] for i in sample_idx]
            sample_refs = [refs[i] for i in sample_idx]
            chrf_samples.append(corpus_chrf(sample_hyps, sample_refs))

        wilson_low, wilson_high = wilson_interval(exact, n)
        rows.append(
            OrderedDict(
                [
                    ("method", name),
                    ("num_examples", n),
                    ("exact_match", exact),
                    ("exact_match_rate", round(exact / n, 4)),
                    ("exact_wilson_low", round(wilson_low, 4)),
                    ("exact_wilson_high", round(wilson_high, 4)),
                    ("corpus_chrf", round(corpus_chrf(hyps, refs), 4)),
                    ("corpus_chrf_ci_low", round(quantile(chrf_samples, 0.025), 4)),
                    ("corpus_chrf_ci_high", round(quantile(chrf_samples, 0.975), 4)),
                ]
            )
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_path.with_suffix(".csv"), rows)
    print(json.dumps(rows, ensure_ascii=False, indent=2))
    print(f"Wrote {output_path}")
    print(f"Wrote {output_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
