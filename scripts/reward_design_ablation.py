#!/usr/bin/env python3
"""Compare alternative offline reward designs on existing DPO pairs.

The goal is diagnostic rather than training-time replacement: quantify how the
current lexical-coverage-based reward compares with simple alternatives that
reuse signals already present in the synthetic multitask targets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import OrderedDict
from difflib import SequenceMatcher
from pathlib import Path

import sacrebleu

import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.lexical_coverage import LexicalCoverageScorer


LITERAL_RE = re.compile(r"<literal>(.*?)</literal>\s*(.*)$", re.S)


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


def parse_targets(output: str) -> tuple[str, str]:
    match = LITERAL_RE.search(output or "")
    if match:
        literal = match.group(1).strip()
        final = match.group(2).strip()
        return literal, final
    return (output or "").strip(), (output or "").strip()


def build_target_lookup(paths: list[Path]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in load_jsonl(path):
            prompt = f"{row['instruction']}\n{row['input']}"
            literal, final = parse_targets(row["output"])
            lookup.setdefault(
                prompt,
                {
                    "literal_target": literal,
                    "final_target": final,
                },
            )
    return lookup


def sentence_chrf(text: str, ref: str) -> float:
    return sacrebleu.sentence_chrf(text, [ref], char_order=6, word_order=2, beta=2).score / 100.0


def sequence_overlap(text: str, ref: str) -> float:
    return SequenceMatcher(None, text, ref).ratio()


def parse_tangut_input(prompt: str) -> str:
    if "\n" in prompt:
        return prompt.split("\n", 1)[1]
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnostic reward-design ablation on Tangut DPO pairs.")
    parser.add_argument("--pairs", default="data/dpo/dpo_pairs.jsonl")
    parser.add_argument("--reward-dict", default="data/dictionary/reward_dict.json")
    parser.add_argument(
        "--train-data",
        nargs="*",
        default=[
            "data/sft/combined_sft_multitask.jsonl",
            "data/sft/combined_sft_unk.jsonl",
            "data/sft/combined_sft.jsonl",
            "data/sft/combined_sft_semantic.jsonl",
        ],
    )
    parser.add_argument("--output", default="results/analysis/reward_design_ablation.json")
    args = parser.parse_args()

    pairs = load_jsonl(Path(args.pairs))
    target_lookup = build_target_lookup([Path(x) for x in args.train_data])
    lex = LexicalCoverageScorer(args.reward_dict)

    variants = OrderedDict(
        [
            ("current_lex_ppl", "Stored repository reward: lexical coverage - 0.01 * log PPL"),
            ("lex_only", "Lexical coverage only"),
            ("literal_chrf", "Sentence chrF++ against multitask <literal> target"),
            ("lex_plus_literal", "0.5 * lexical coverage + 0.5 * literal chrF++"),
            ("oracle_final_chrf", "Diagnostic oracle: sentence chrF++ against final synthetic target"),
        ]
    )

    summary_rows = []
    for variant, description in variants.items():
        chosen_better = 0
        rejected_better = 0
        equal = 0
        kept_02 = 0
        kept_04 = 0
        kept_02_good = 0
        kept_04_good = 0
        sign_flips = 0
        mean_gap = 0.0
        matched = 0
        finite_gap_count = 0

        for row in pairs:
            prompt = row["prompt"]
            targets = target_lookup.get(prompt)
            if targets is None:
                continue

            tangut_input = parse_tangut_input(prompt)
            chosen = row["chosen"]
            rejected = row["rejected"]
            literal_target = targets["literal_target"]
            final_target = targets["final_target"]

            chosen_lex = lex.score(tangut_input, chosen)
            rejected_lex = lex.score(tangut_input, rejected)
            chosen_literal = sentence_chrf(chosen, literal_target)
            rejected_literal = sentence_chrf(rejected, literal_target)
            chosen_final = sentence_chrf(chosen, final_target)
            rejected_final = sentence_chrf(rejected, final_target)

            if variant == "current_lex_ppl":
                chosen_reward = row["chosen_reward"]
                rejected_reward = row["rejected_reward"]
            elif variant == "lex_only":
                chosen_reward = chosen_lex
                rejected_reward = rejected_lex
            elif variant == "literal_chrf":
                chosen_reward = chosen_literal
                rejected_reward = rejected_literal
            elif variant == "lex_plus_literal":
                chosen_reward = 0.5 * chosen_lex + 0.5 * chosen_literal
                rejected_reward = 0.5 * rejected_lex + 0.5 * rejected_literal
            elif variant == "oracle_final_chrf":
                chosen_reward = chosen_final
                rejected_reward = rejected_final
            else:
                raise ValueError(f"Unsupported variant: {variant}")

            current_gap = row["chosen_reward"] - row["rejected_reward"]
            gap = chosen_reward - rejected_reward
            matched += 1
            if math.isfinite(gap):
                mean_gap += gap
                finite_gap_count += 1

            if math.isfinite(current_gap) and math.isfinite(gap) and ((gap >= 0) != (current_gap >= 0)):
                sign_flips += 1

            chosen_seq = sequence_overlap(chosen, final_target)
            rejected_seq = sequence_overlap(rejected, final_target)
            if chosen_seq > rejected_seq:
                chosen_better += 1
                good = True
            elif rejected_seq > chosen_seq:
                rejected_better += 1
                good = False
            else:
                equal += 1
                good = None

            if gap >= 0.2:
                kept_02 += 1
                if good is True:
                    kept_02_good += 1
            if gap >= 0.4:
                kept_04 += 1
                if good is True:
                    kept_04_good += 1

        summary_rows.append(
            OrderedDict(
                [
                    ("variant", variant),
                    ("description", description),
                    ("matched_pairs", matched),
                    ("chosen_better_rate_vs_final", round(chosen_better / matched, 4)),
                    ("rejected_better_rate_vs_final", round(rejected_better / matched, 4)),
                    ("equal_rate_vs_final", round(equal / matched, 4)),
                    (
                        "mean_reward_gap",
                        round(mean_gap / finite_gap_count, 4) if finite_gap_count else None,
                    ),
                    ("pairs_kept_ge_0_2", kept_02),
                    ("pairs_kept_ge_0_4", kept_04),
                    (
                        "kept_ge_0_2_good_rate_vs_final",
                        round(kept_02_good / kept_02, 4) if kept_02 else None,
                    ),
                    (
                        "kept_ge_0_4_good_rate_vs_final",
                        round(kept_04_good / kept_04, 4) if kept_04 else None,
                    ),
                    ("sign_flips_vs_current", sign_flips),
                ]
            )
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(output_path.with_suffix(".csv"), summary_rows)
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    print(f"Wrote {output_path}")
    print(f"Wrote {output_path.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
