#!/usr/bin/env python3
"""Filter DPO pairs by reward gap and basic validity checks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter Tangut-NLP DPO pairs.")
    parser.add_argument("--input", default="data/dpo/dpo_pairs.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-gap", type=float, default=0.2)
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Drop exact duplicate chosen/rejected text pairs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary only; do not write output.",
    )
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    kept = []
    seen = set()

    stats = {
        "input_rows": len(rows),
        "nonfinite": 0,
        "duplicate_same_text": 0,
        "below_gap": 0,
        "deduped": 0,
    }

    for row in rows:
        chosen_reward = row.get("chosen_reward")
        rejected_reward = row.get("rejected_reward")
        if not math.isfinite(chosen_reward) or not math.isfinite(rejected_reward):
            stats["nonfinite"] += 1
            continue
        if row.get("chosen") == row.get("rejected"):
            stats["duplicate_same_text"] += 1
            continue

        gap = chosen_reward - rejected_reward
        if gap < args.min_gap:
            stats["below_gap"] += 1
            continue

        if args.dedupe:
            key = (row.get("prompt"), row.get("chosen"), row.get("rejected"))
            if key in seen:
                stats["deduped"] += 1
                continue
            seen.add(key)

        kept.append(row)

    stats["kept"] = len(kept)
    stats["keep_rate"] = (len(kept) / len(rows)) if rows else 0.0

    print(json.dumps(stats, ensure_ascii=False, indent=2))

    if args.dry_run:
        return

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
