#!/usr/bin/env python3
"""Simple open reranking baselines over an existing Tangut candidate pool.

This script addresses reviewer requests for standard open reranking controls.
It reconstructs a candidate pool from existing prediction files and evaluates:

1. plurality vote: choose the most frequent string in the pool
2. pairwise chrF++ MBR: choose the candidate with highest average agreement
   with the rest of the pool

Both baselines are fully offline and never call an external model.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, OrderedDict
from pathlib import Path

import sacrebleu

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.open_hybrid_heuristic import BAD_PATTERN, load_jsonl  # noqa: E402


DIAG_TITLE_SUFFIXES = set("經論記疏頌儀義傳錄贊序字品")
DEFAULT_PREDICTIONS = {
    "frontier": REPO_ROOT / "results/frontier_deepseek_v32_fewshot_cot/predictions.jsonl",
    "unk": REPO_ROOT / "results/baseline3_1_unk/predictions.jsonl",
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


def sentence_chrf(hyp: str, ref: str) -> float:
    return sacrebleu.sentence_chrf(hyp, [ref], char_order=6, word_order=2, beta=2).score


def load_prediction_map(overrides: list[str]) -> dict[str, Path]:
    pred_map = dict(DEFAULT_PREDICTIONS)
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=PATH, got {spec}")
        name, raw_path = spec.split("=", 1)
        pred_map[name.strip()] = Path(raw_path).expanduser()
    return pred_map


def build_candidate_pool(rows_by_name: dict[str, list[dict]], idx: int) -> tuple[str, str, str, list[dict]]:
    source = None
    reference = None
    glosses = None
    candidates = []
    for name, rows in rows_by_name.items():
        row = rows[idx]
        if source is None:
            source = row["input"]
            reference = row.get("reference", "")
            glosses = row.get("glosses", "")
        elif row["input"] != source:
            raise ValueError(f"Input mismatch at row {idx + 1}: {name}")
        prediction = (row.get("prediction", "") or "").strip()
        if not prediction:
            continue
        candidates.append({"name": name, "prediction": prediction})
    if not candidates:
        raise ValueError(f"No usable candidates at row {idx + 1}")
    return source or "", reference or "", glosses or "", candidates


def choose_vote(candidates: list[dict], frontier_name: str) -> tuple[dict, list[dict]]:
    counts = Counter(item["prediction"] for item in candidates)
    top_count = max(counts.values())
    top_strings = {pred for pred, count in counts.items() if count == top_count}

    frontier = next((item for item in candidates if item["name"] == frontier_name), None)
    if frontier and frontier["prediction"] in top_strings:
        chosen = frontier
    else:
        chosen = next(item for item in candidates if item["prediction"] in top_strings)

    diagnostics = []
    for item in candidates:
        diagnostics.append(
            {
                "name": item["name"],
                "prediction": item["prediction"],
                "vote_count": counts[item["prediction"]],
            }
        )
    return chosen, diagnostics


def choose_pairwise_mbr(candidates: list[dict], frontier_name: str) -> tuple[dict, list[dict]]:
    scored = []
    for i, item in enumerate(candidates):
        if len(candidates) == 1:
            mean_agreement = 100.0
        else:
            agreements = []
            for j, other in enumerate(candidates):
                if i == j:
                    continue
                agreements.append(sentence_chrf(item["prediction"], other["prediction"]))
            mean_agreement = sum(agreements) / len(agreements)
        scored.append(
            {
                "name": item["name"],
                "prediction": item["prediction"],
                "mbr_agreement": mean_agreement,
            }
        )

    max_score = max(item["mbr_agreement"] for item in scored)
    top = [item for item in scored if item["mbr_agreement"] == max_score]
    chosen = next((item for item in top if item["name"] == frontier_name), top[0])
    return chosen, scored


def representative_basis(candidates: list[dict], chosen_prediction: str, frontier_name: str) -> str:
    frontier = next((item for item in candidates if item["name"] == frontier_name), None)
    if frontier and frontier["prediction"] == chosen_prediction:
        return frontier_name
    for item in candidates:
        if item["prediction"] == chosen_prediction:
            return item["name"]
    return candidates[0]["name"]


def build_rows(
    *,
    rows_by_name: dict[str, list[dict]],
    mode: str,
    frontier_name: str,
) -> list[dict]:
    num_rows = len(next(iter(rows_by_name.values())))
    output_rows = []
    for idx in range(num_rows):
        source, reference, glosses, candidates = build_candidate_pool(rows_by_name, idx)
        if mode == "vote":
            chosen, diagnostics = choose_vote(candidates, frontier_name)
        elif mode == "pairwise_mbr":
            chosen, diagnostics = choose_pairwise_mbr(candidates, frontier_name)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        output_rows.append(
            {
                "input": source,
                "reference": reference,
                "prediction": chosen["prediction"],
                "glosses": glosses,
                "method": f"{mode}_open_rerank",
                "selector_basis": representative_basis(candidates, chosen["prediction"], frontier_name),
                "candidate_scores": diagnostics,
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

    basis_counts = Counter(row["selector_basis"] for row in rows)
    switched_from_frontier = sum(int(row["selector_basis"] != frontier_name) for row in rows)

    return OrderedDict(
        [
            ("num_examples", len(rows)),
            ("corpus_chrf", round(corpus_chrf(hyps, refs), 4)),
            ("exact_match", exact),
            ("exact_match_rate", round(exact / len(rows), 4)),
            ("contamination_rate", round(contamination / len(rows), 4)),
            (
                "length_ratio",
                round(sum(len(h) for h in hyps) / max(1, sum(len(r) for r in refs)), 4),
            ),
            ("title_suffix_ok", suffix_ok),
            ("title_suffix_total", suffix_total),
            ("switched_from_frontier", switched_from_frontier),
            ("basis_counts", json.dumps(dict(sorted(basis_counts.items())), ensure_ascii=False)),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run simple open reranking baselines over an existing candidate pool.")
    parser.add_argument("--pred", action="append", default=[], help="Optional NAME=PATH override for prediction files.")
    parser.add_argument(
        "--mode",
        choices=["vote", "pairwise_mbr", "all"],
        default="all",
    )
    parser.add_argument("--frontier-name", default="frontier")
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "results/analysis/candidate_pool_baselines"),
    )
    args = parser.parse_args()

    pred_map = load_prediction_map(args.pred)
    rows_by_name = {name: load_jsonl(path) for name, path in pred_map.items()}
    lengths = {len(rows) for rows in rows_by_name.values()}
    if len(lengths) != 1:
        raise ValueError(f"Prediction length mismatch: {lengths}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    modes = ["vote", "pairwise_mbr"] if args.mode == "all" else [args.mode]
    summary_rows = []
    for mode in modes:
        rows = build_rows(rows_by_name=rows_by_name, mode=mode, frontier_name=args.frontier_name)
        metrics = evaluate_predictions(rows, args.frontier_name)
        summary = OrderedDict([("mode", mode)])
        summary.update(metrics)
        summary_rows.append(summary)

        pred_path = output_dir / f"{mode}.jsonl"
        with pred_path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    summary_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(summary_csv, summary_rows)

    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
