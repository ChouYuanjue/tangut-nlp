#!/usr/bin/env python3
"""Deterministic contamination and failure taxonomy audit for Tangut titles.

This script is designed for reviewer-facing diagnostics over stored predictions.
It keeps the definitions deliberately simple and reproducible:

1. contamination is a binary artifact flag with no tuned threshold;
2. non-exact outputs are assigned a primary failure label in a fixed order.

The goal is not to replace human judgment, but to make the paper's
"contamination" narrative operational and to add a lightweight error taxonomy.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import OrderedDict
from pathlib import Path


TITLE_SUFFIXES = set("經論記疏頌儀義傳錄贊序字品文觀根次門")
NUMERIC_CHARS = set("零〇一二三四五六七八九十百千萬兩")
PROMPT_ARTIFACT_LEXEMES = (
    "assistant",
    "manuals",
    "networks",
    "shows",
    "grat",
    "dresses",
)

ASCII_ALPHA_RE = re.compile(r"[A-Za-z]")
ANGLE_OR_TAG_RE = re.compile(r"[<>]")
UNK_RE = re.compile(r"\[UNK\]", re.IGNORECASE)
PROMPT_ARTIFACT_RE = re.compile("|".join(PROMPT_ARTIFACT_LEXEMES), re.IGNORECASE)

DEFAULT_METHODS = [
    "frontier_deepseek_v32_fewshot_cot",
    "baseline3_2_multitask",
    "final_gap04_multitask_sigmoid",
    "open_hybrid_heuristic_guarded",
    "hybrid_multi3_catalog_gpt54",
    "final_v2",
]

DISPLAY_NAMES = {
    "frontier_deepseek_v32_fewshot_cot": "Frontier DeepSeek",
    "baseline3_2_multitask": "MT-SFT",
    "final_gap04_multitask_sigmoid": "MT-DPO g0.4 sig",
    "open_hybrid_heuristic_guarded": "Open selector",
    "hybrid_multi3_catalog_gpt54": "3-way catalog adjudicator",
    "final_v2": "Legacy DPO",
}


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


def prediction_path(results_dir: Path, method: str) -> Path:
    custom = {
        "baseline3_2_multitask": results_dir / method / "predictions_cleaned.jsonl",
        "open_hybrid_heuristic_guarded": results_dir / method / "predictions.jsonl",
    }
    return custom.get(method, results_dir / method / "predictions.jsonl")


def contamination_flags(prediction: str) -> dict[str, int]:
    return {
        "ascii_alpha": int(bool(ASCII_ALPHA_RE.search(prediction))),
        "angle_or_tag": int(bool(ANGLE_OR_TAG_RE.search(prediction))),
        "unk_token": int(bool(UNK_RE.search(prediction))),
        "prompt_artifact": int(bool(PROMPT_ARTIFACT_RE.search(prediction))),
    }


def numeric_signature(text: str) -> str:
    return "".join(ch for ch in text if ch in NUMERIC_CHARS)


def error_labels(prediction: str, reference: str) -> dict[str, object]:
    prediction = (prediction or "").strip()
    reference = (reference or "").strip()

    if prediction == reference:
        return {
            "primary_error": "exact",
            "all_error_labels": ["exact"],
            "title_suffix_miss": 0,
            "number_mismatch": 0,
            "truncation": 0,
            "over_expansion": 0,
        }

    flags = contamination_flags(prediction)
    labels = []
    if any(flags.values()):
        labels.append("contamination")

    truncation = int(bool(reference) and len(prediction) <= max(1, len(reference) // 2))
    over_expansion = int(
        len(prediction) >= len(reference) + 4 and len(prediction) * 2 >= len(reference) * 3
    )
    if truncation:
        labels.append("truncation")
    if over_expansion:
        labels.append("over_expansion")

    ref_suffixes = {ch for ch in reference if ch in TITLE_SUFFIXES}
    title_suffix_miss = int(bool(ref_suffixes) and not all(ch in prediction for ch in ref_suffixes))
    if title_suffix_miss:
        labels.append("suffix_drop")

    ref_nums = numeric_signature(reference)
    pred_nums = numeric_signature(prediction)
    number_mismatch = int(bool(ref_nums) and bool(pred_nums) and ref_nums != pred_nums)
    if number_mismatch:
        labels.append("number_mismatch")

    primary_error = next(
        (
            label
            for label in [
                "contamination",
                "truncation",
                "over_expansion",
                "suffix_drop",
                "number_mismatch",
            ]
            if label in labels
        ),
        "content_drift",
    )
    if primary_error == "content_drift":
        labels.append("content_drift")

    return {
        "primary_error": primary_error,
        "all_error_labels": labels,
        "title_suffix_miss": title_suffix_miss,
        "number_mismatch": number_mismatch,
        "truncation": truncation,
        "over_expansion": over_expansion,
        **flags,
    }


def summarize_method(
    *,
    method: str,
    rows: list[dict],
    test_rows: list[dict],
) -> tuple[OrderedDict, list[OrderedDict]]:
    summary = OrderedDict(
        [
            ("method", method),
            ("display_name", DISPLAY_NAMES.get(method, method)),
            ("num_examples", len(test_rows)),
            ("exact", 0),
            ("primary_contamination", 0),
            ("primary_truncation", 0),
            ("primary_over_expansion", 0),
            ("primary_suffix_drop", 0),
            ("primary_number_mismatch", 0),
            ("primary_suffix_or_number", 0),
            ("primary_content_drift", 0),
            ("flag_contamination", 0),
            ("flag_ascii_alpha", 0),
            ("flag_angle_or_tag", 0),
            ("flag_unk_token", 0),
            ("flag_prompt_artifact", 0),
            ("flag_truncation", 0),
            ("flag_over_expansion", 0),
            ("flag_suffix_drop", 0),
            ("flag_number_mismatch", 0),
            ("flag_suffix_or_number", 0),
        ]
    )

    detail_rows: list[OrderedDict] = []
    for idx, (prediction_row, test_row) in enumerate(zip(rows, test_rows), start=1):
        prediction = (prediction_row.get("prediction", "") or "").strip()
        reference = test_row["output"]
        labels = error_labels(prediction, reference)
        primary_error = labels["primary_error"]

        if primary_error == "exact":
            summary["exact"] += 1
        else:
            summary[f"primary_{primary_error}"] += 1

        contamination = int(
            labels.get("ascii_alpha", 0)
            or labels.get("angle_or_tag", 0)
            or labels.get("unk_token", 0)
            or labels.get("prompt_artifact", 0)
        )
        summary["flag_contamination"] += contamination
        summary["flag_ascii_alpha"] += labels.get("ascii_alpha", 0)
        summary["flag_angle_or_tag"] += labels.get("angle_or_tag", 0)
        summary["flag_unk_token"] += labels.get("unk_token", 0)
        summary["flag_prompt_artifact"] += labels.get("prompt_artifact", 0)
        summary["flag_truncation"] += labels["truncation"]
        summary["flag_over_expansion"] += labels["over_expansion"]
        summary["flag_suffix_drop"] += labels["title_suffix_miss"]
        summary["flag_number_mismatch"] += labels["number_mismatch"]

        detail_rows.append(
            OrderedDict(
                [
                    ("method", method),
                    ("display_name", DISPLAY_NAMES.get(method, method)),
                    ("index", idx),
                    ("input", test_row["input"]),
                    ("reference", reference),
                    ("prediction", prediction),
                    ("is_exact", int(primary_error == "exact")),
                    ("primary_error", primary_error),
                    ("all_error_labels", json.dumps(labels["all_error_labels"], ensure_ascii=False)),
                    ("ascii_alpha", labels.get("ascii_alpha", 0)),
                    ("angle_or_tag", labels.get("angle_or_tag", 0)),
                    ("unk_token", labels.get("unk_token", 0)),
                    ("prompt_artifact", labels.get("prompt_artifact", 0)),
                    ("truncation", labels["truncation"]),
                    ("over_expansion", labels["over_expansion"]),
                    ("suffix_drop", labels["title_suffix_miss"]),
                    ("number_mismatch", labels["number_mismatch"]),
                ]
            )
        )

    summary["primary_suffix_or_number"] = (
        summary["primary_suffix_drop"] + summary["primary_number_mismatch"]
    )
    summary["flag_suffix_or_number"] = (
        summary["flag_suffix_drop"] + summary["flag_number_mismatch"]
    )
    return summary, detail_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit contamination and deterministic failure taxonomy on Tangut titles."
    )
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--test-set", default="data/eval/test_set.jsonl")
    parser.add_argument("--output-dir", default="results/analysis/title_diagnostics")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    test_rows = load_jsonl(Path(args.test_set))

    summary_rows = []
    detail_rows = []
    for method in args.methods:
        pred_rows = load_jsonl(prediction_path(results_dir, method))
        if len(pred_rows) != len(test_rows):
            raise ValueError(
                f"Prediction/reference length mismatch for {method}: "
                f"{len(pred_rows)} vs {len(test_rows)}"
            )
        summary, details = summarize_method(method=method, rows=pred_rows, test_rows=test_rows)
        summary_rows.append(summary)
        detail_rows.extend(details)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    details_csv = output_dir / "details.csv"

    summary_json.write_text(
        json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(summary_csv, summary_rows)
    write_csv(details_csv, detail_rows)

    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")
    print(f"Wrote {details_csv}")


if __name__ == "__main__":
    main()
