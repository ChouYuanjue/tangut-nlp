#!/usr/bin/env python3
"""Analyze sensitivity of DPO pair quality to reward weighting.

This script recomputes the repository reward

    lex(y) - w * log ppl(y)

for several weights ``w`` and reports:
1. how many pair orderings flip relative to the current reward,
2. how many pairs survive gap filtering at 0.2 / 0.4, and
3. how often the reward-chosen candidate is closer to a simple gold proxy.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
from difflib import SequenceMatcher
from pathlib import Path

from eval.lexical_coverage import LexicalCoverageScorer
from eval.perplexity import PerplexityScorer


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_train_lookup(paths: list[Path]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        for row in load_jsonl(path):
            lookup[f"{row['instruction']}\n{row['input']}"] = row["output"]
    return lookup


def parse_tangut_input(prompt: str) -> str:
    if "\n" in prompt:
        return prompt.split("\n", 1)[1]
    return prompt


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate the reward weight in Tangut DPO pair construction.")
    parser.add_argument("--pairs", default="data/dpo/dpo_pairs.jsonl")
    parser.add_argument("--reward-dict", default="data/dictionary/reward_dict.json")
    parser.add_argument("--ppl-model", default="models/qwen2.5-0.5b")
    parser.add_argument(
        "--train-data",
        nargs="*",
        default=[
            "data/sft/combined_sft_unk.jsonl",
            "data/sft/combined_sft.jsonl",
            "data/sft/combined_sft_semantic.jsonl",
            "data/sft/combined_sft_multitask.jsonl",
        ],
    )
    parser.add_argument("--weights", nargs="*", type=float, default=[0.0, 0.005, 0.01, 0.02])
    parser.add_argument("--output", default="results/analysis/reward_weight_ablation.json")
    args = parser.parse_args()

    pairs = load_jsonl(Path(args.pairs))
    train_lookup = build_train_lookup([Path(x) for x in args.train_data])

    lex = LexicalCoverageScorer(args.reward_dict)
    ppl = PerplexityScorer(args.ppl_model)

    all_texts = sorted({row["chosen"] for row in pairs} | {row["rejected"] for row in pairs})
    ppl_cache: dict[str, float] = {}
    for idx, text in enumerate(all_texts, start=1):
        ppl_cache[text] = ppl.score(text)
        if idx % 500 == 0:
            print(f"[ppl] scored {idx}/{len(all_texts)} unique texts")

    results: OrderedDict[str, dict] = OrderedDict()
    for weight in args.weights:
        found = 0
        chosen_better = 0
        rejected_better = 0
        equal = 0
        kept_02 = 0
        kept_04 = 0
        kept_02_good = 0
        kept_04_good = 0
        sign_flips = 0

        for row in pairs:
            tangut_input = parse_tangut_input(row["prompt"])
            chosen_lex = lex.score(tangut_input, row["chosen"])
            rejected_lex = lex.score(tangut_input, row["rejected"])
            chosen_reward = chosen_lex - weight * math.log(max(ppl_cache[row["chosen"]], 1e-9))
            rejected_reward = rejected_lex - weight * math.log(max(ppl_cache[row["rejected"]], 1e-9))
            gap = chosen_reward - rejected_reward
            orig_gap = row["chosen_reward"] - row["rejected_reward"]
            if (gap >= 0) != (orig_gap >= 0):
                sign_flips += 1

            gold = train_lookup.get(row["prompt"])
            good = None
            if gold is not None:
                found += 1
                chosen_sim = SequenceMatcher(None, row["chosen"], gold).ratio()
                rejected_sim = SequenceMatcher(None, row["rejected"], gold).ratio()
                if chosen_sim > rejected_sim:
                    chosen_better += 1
                    good = True
                elif rejected_sim > chosen_sim:
                    rejected_better += 1
                    good = False
                else:
                    equal += 1

            if gap >= 0.2:
                kept_02 += 1
                if good is True:
                    kept_02_good += 1
            if gap >= 0.4:
                kept_04 += 1
                if good is True:
                    kept_04_good += 1

        results[str(weight)] = {
            "chosen_better_rate": (chosen_better / found) if found else None,
            "rejected_better_rate": (rejected_better / found) if found else None,
            "equal_rate": (equal / found) if found else None,
            "kept_ge_0_2": kept_02,
            "kept_ge_0_4": kept_04,
            "kept_ge_0_2_good_rate": (kept_02_good / kept_02) if kept_02 else None,
            "kept_ge_0_4_good_rate": (kept_04_good / kept_04) if kept_04 else None,
            "sign_flips_vs_current": sign_flips,
        }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
