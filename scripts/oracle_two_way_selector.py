#!/usr/bin/env python3
"""Build a conservative 2-way selector over oracle prompt and SFT outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


BAD_PATTERN = re.compile(r"[A-Za-z<>]|\[UNK\]|\s")


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def is_subsequence(shorter: str, longer: str) -> bool:
    cursor = iter(longer)
    return all(ch in cursor for ch in shorter)


def choose_prediction(dict_pred: str, sft_pred: str) -> tuple[str, str]:
    dict_bad = bool(BAD_PATTERN.search(dict_pred))
    sft_bad = bool(BAD_PATTERN.search(sft_pred))

    if dict_bad and not sft_bad:
        return "sft", "dict_bad"
    if sft_bad and not dict_bad:
        return "dict", "sft_bad"

    # Conservative repair rule: allow SFT to override only if it looks like
    # a short clean completion of the prompt-based candidate.
    if len(sft_pred) > len(dict_pred) and len(sft_pred) - len(dict_pred) <= 3:
        if is_subsequence(dict_pred, sft_pred):
            return "sft", "sft_supersequence"

    if len(dict_pred) > len(sft_pred) and len(dict_pred) - len(sft_pred) <= 3:
        if is_subsequence(sft_pred, dict_pred):
            return "dict", "dict_supersequence"

    return "dict", "default_dict"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a conservative 2-way oracle selector over dict prompt and MT-SFT literal outputs."
    )
    parser.add_argument("--dict-predictions", required=True)
    parser.add_argument("--sft-predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--analysis-output", required=True)
    parser.add_argument("--method-name", default="oracle_two_way_selector")
    args = parser.parse_args()

    dict_rows = load_jsonl(Path(args.dict_predictions))
    sft_rows = load_jsonl(Path(args.sft_predictions))
    if len(dict_rows) != len(sft_rows):
        raise ValueError(
            f"Prediction length mismatch: {len(dict_rows)} vs {len(sft_rows)}"
        )

    chosen_rows: list[dict] = []
    analysis_rows: list[dict] = []

    for idx, (dict_row, sft_row) in enumerate(zip(dict_rows, sft_rows), start=1):
        if dict_row.get("input") != sft_row.get("input"):
            raise ValueError(f"Input mismatch at row {idx}")

        choice, reason = choose_prediction(
            dict_pred=dict_row.get("prediction", ""),
            sft_pred=sft_row.get("prediction", ""),
        )
        chosen = dict_row if choice == "dict" else sft_row

        chosen_rows.append(
            {
                "input": chosen.get("input", ""),
                "reference": chosen.get("reference", ""),
                "prediction": chosen.get("prediction", ""),
                "method": args.method_name,
            }
        )
        analysis_rows.append(
            {
                "index": idx,
                "choice": choice,
                "reason": reason,
                "dict_prediction": dict_row.get("prediction", ""),
                "sft_prediction": sft_row.get("prediction", ""),
                "chosen_prediction": chosen.get("prediction", ""),
            }
        )

    dump_jsonl(Path(args.output), chosen_rows)
    dump_jsonl(Path(args.analysis_output), analysis_rows)
    print(
        json.dumps(
            {
                "status": "ok",
                "rows": len(chosen_rows),
                "output": args.output,
                "analysis_output": args.analysis_output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
