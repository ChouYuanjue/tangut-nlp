"""Deterministic open-weight reranking baseline for Tangut title candidates.

This script implements a conservative, fully reproducible selector over an
existing candidate pool. It does not call any external API and does not
generate new text. Instead, it scores candidates using cheap signals that are
available at inference time:

1. pairwise prefix/suffix consensus across candidates
2. title-suffix prior
3. short-title anti-truncation / anti-expansion guards
4. contamination penalties
5. a small frontier prior to avoid over-switching

The goal is not to beat the catalog-aware LLM adjudicator, but to provide a
transparent sanity baseline for reviewer questions about simpler open rerankers.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


TITLE_SUFFIXES = set("經論記疏頌儀義傳錄贊序字品文觀根次門")
BAD_PATTERN = re.compile(
    r"[A-Za-z<>]|\[UNK\]|assistant|manuals|networks|shows|grat|dresses",
    re.IGNORECASE,
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def common_prefix(a: str, b: str) -> str:
    out = []
    for ch_a, ch_b in zip(a, b):
        if ch_a != ch_b:
            break
        out.append(ch_a)
    return "".join(out)


def common_suffix(a: str, b: str) -> str:
    out = []
    for ch_a, ch_b in zip(reversed(a), reversed(b)):
        if ch_a != ch_b:
            break
        out.append(ch_a)
    return "".join(reversed(out))


def candidate_score(
    *,
    pred: str,
    others: list[str],
    source_length: int,
    median_length: int,
    is_frontier: bool,
) -> tuple[float, dict]:
    prefix_consensus = 0
    suffix_consensus = 0
    for other in others:
        prefix = common_prefix(pred, other)
        suffix = common_suffix(pred, other)
        if len(prefix) >= 2:
            prefix_consensus += len(prefix)
        if len(suffix) >= 2 and suffix != prefix:
            suffix_consensus += len(suffix)

    has_title_suffix = int(bool(pred) and pred[-1] in TITLE_SUFFIXES)
    contamination = int(bool(BAD_PATTERN.search(pred)))
    too_short = int(len(pred) <= max(1, source_length // 4))
    too_long = int(len(pred) >= median_length + 5 and len(pred) >= source_length + 5)
    length_penalty = abs(len(pred) - median_length)

    score = 0.0
    score += 1.4 * prefix_consensus
    score += 1.2 * suffix_consensus
    score += 2.5 * has_title_suffix
    score -= 1.0 * length_penalty
    score -= 6.0 * contamination
    score -= 4.0 * too_short
    score -= 2.0 * too_long
    if is_frontier:
        score += 0.75

    diagnostics = {
        "prefix_consensus": prefix_consensus,
        "suffix_consensus": suffix_consensus,
        "has_title_suffix": has_title_suffix,
        "contamination": contamination,
        "too_short": too_short,
        "too_long": too_long,
        "length_penalty": length_penalty,
        "frontier_prior": 0.75 if is_frontier else 0.0,
    }
    return score, diagnostics


def choose_guarded(scored: list[dict], frontier_name: str) -> dict:
    frontier = next((item for item in scored if item["name"] == frontier_name), None)
    if frontier is None:
        return max(scored, key=lambda x: (x["score"], x["name"] == frontier_name))

    locals_only = [item for item in scored if item["name"] != frontier_name]
    if not locals_only:
        return frontier

    best_local = max(locals_only, key=lambda x: x["score"])
    fd = frontier["diagnostics"]
    ld = best_local["diagnostics"]

    # Hard guards first: obvious truncation or contamination in the frontier.
    if fd["contamination"] and not ld["contamination"]:
        return best_local
    if len(frontier["prediction"]) <= 1 and len(best_local["prediction"]) >= 3:
        return best_local

    frontier_consensus = fd["prefix_consensus"] + fd["suffix_consensus"]
    local_consensus = ld["prefix_consensus"] + ld["suffix_consensus"]
    local_margin = best_local["score"] - frontier["score"]

    # Switch only when the local option has materially stronger conservative evidence.
    if (
        local_margin >= 3.0
        and local_consensus >= frontier_consensus + 2
        and ld["too_short"] == 0
        and ld["contamination"] == 0
        and (ld["has_title_suffix"] or not fd["has_title_suffix"])
    ):
        return best_local

    # Also allow a switch when the frontier is clearly over-expanded while the local
    # candidate is shorter and supported by consensus.
    if (
        fd["too_long"]
        and not ld["too_long"]
        and local_consensus >= 2
        and len(best_local["prediction"]) + 2 <= len(frontier["prediction"])
    ):
        return best_local

    return frontier


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic heuristic reranking over candidate pools.")
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        help="Repeated NAME=PATH spec for candidate prediction files.",
    )
    parser.add_argument("--output", required=True, help="Path to save reranked predictions.jsonl.")
    parser.add_argument("--frontier-name", default="frontier", help="Candidate name treated as the default frontier.")
    parser.add_argument(
        "--mode",
        choices=["guarded", "score_max"],
        default="guarded",
        help="Selection strategy: guarded frontier-first switch or raw score maximization.",
    )
    args = parser.parse_args()

    rows_by_name: dict[str, list[dict]] = {}
    for spec in args.pred:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=PATH, got: {spec}")
        name, raw_path = spec.split("=", 1)
        rows_by_name[name.strip()] = load_jsonl(Path(raw_path).expanduser())

    lengths = {len(v) for v in rows_by_name.values()}
    if len(lengths) != 1:
        raise ValueError(f"Prediction length mismatch: {lengths}")

    output_rows = []
    num_rows = next(iter(lengths))
    for idx in range(num_rows):
        candidates = []
        source = None
        reference = None
        glosses = None
        for name, rows in rows_by_name.items():
            row = rows[idx]
            if source is None:
                source = row["input"]
                reference = row.get("reference", "")
                glosses = row.get("glosses", "")
            elif row["input"] != source:
                raise ValueError(f"Input mismatch at row {idx + 1}: {name}")
            pred = (row.get("prediction", "") or "").strip()
            if not pred:
                continue
            candidates.append({"name": name, "prediction": pred})

        preds = [c["prediction"] for c in candidates]
        median_length = sorted(len(x) for x in preds)[len(preds) // 2]
        scored = []
        for cand in candidates:
            others = [x for x in preds if x != cand["prediction"]]
            score, diagnostics = candidate_score(
                pred=cand["prediction"],
                others=others,
                source_length=len(source or ""),
                median_length=median_length,
                is_frontier=(cand["name"] == args.frontier_name),
            )
            scored.append(
                {
                    "name": cand["name"],
                    "prediction": cand["prediction"],
                    "score": score,
                    "diagnostics": diagnostics,
                }
            )

        if args.mode == "guarded":
            best = choose_guarded(scored, args.frontier_name)
        else:
            best = max(scored, key=lambda x: (x["score"], x["name"] == args.frontier_name))
        output_rows.append(
            {
                "input": source,
                "reference": reference,
                "prediction": best["prediction"],
                "glosses": glosses,
                "method": f"open_hybrid_heuristic_{args.mode}",
                "selector_basis": best["name"],
                "selector_score": round(best["score"], 4),
                "selector_mode": args.mode,
                "selector_diagnostics": best["diagnostics"],
                "candidate_scores": [
                    {
                        "name": item["name"],
                        "score": round(item["score"], 4),
                        **item["diagnostics"],
                    }
                    for item in scored
                ],
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    basis_counts: dict[str, int] = {}
    for row in output_rows:
        basis_counts[row["selector_basis"]] = basis_counts.get(row["selector_basis"], 0) + 1
    print(json.dumps({"num_examples": len(output_rows), "basis_counts": basis_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
