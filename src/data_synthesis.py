"""Generate synthetic Tangut-Chinese parallel corpus from ancient Chinese texts."""

import argparse
import json
import random
from pathlib import Path


def build_cn_to_tangut_map(dictionary_path):
    with open(dictionary_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    cn_to_tangut = {}
    for entry in entries:
        tangut_char = entry.get("character", "")
        if len(tangut_char) != 1:
            continue
        cn_explanation = entry.get("explanationCN", "").strip()
        if not cn_explanation:
            continue
        for c in cn_explanation:
            if "\u4e00" <= c <= "\u9fff":
                if c not in cn_to_tangut:
                    cn_to_tangut[c] = []
                if tangut_char not in cn_to_tangut[c]:
                    cn_to_tangut[c].append(tangut_char)
    return cn_to_tangut


def synthesize_one_pair(ancient_text, modern_text, cn_to_tangut, replacement_ratio):
    chars = list(ancient_text)
    replaceable_indices = [i for i, c in enumerate(chars) if c in cn_to_tangut]
    if len(replaceable_indices) < 2:
        return None
    num_to_replace = max(1, int(len(replaceable_indices) * replacement_ratio))
    indices_to_replace = random.sample(replaceable_indices, min(num_to_replace, len(replaceable_indices)))
    for idx in indices_to_replace:
        chars[idx] = random.choice(cn_to_tangut[chars[idx]])
    return "".join(chars), modern_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dictionary-path", default="data/dictionary/dictionary.json")
    parser.add_argument("--ancient-chinese-path", default="data/raw/ancient_chinese_hf")
    parser.add_argument("--output", default="data/sft/synthetic_sft.jsonl")
    parser.add_argument("--max-samples", type=int, default=50000)
    parser.add_argument("--min-length", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=64)
    parser.add_argument("--replacement-min", type=float, default=0.3)
    parser.add_argument("--replacement-max", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    cn_to_tangut = build_cn_to_tangut_map(args.dictionary_path)
    print(f"CN->Tangut map: {len(cn_to_tangut)} Chinese chars covered")

    from datasets import load_from_disk
    ds = load_from_disk(args.ancient_chinese_path)

    samples = []
    for item in ds:
        if len(samples) >= args.max_samples:
            break
        ancient = item.get("ancient", "") or item.get("source", "")
        modern = item.get("modern", "") or item.get("target", "")
        if not ancient or not modern:
            continue
        if len(ancient) < args.min_length or len(ancient) > args.max_length:
            continue
        ratio = random.uniform(args.replacement_min, args.replacement_max)
        result = synthesize_one_pair(ancient, modern, cn_to_tangut, ratio)
        if result is None:
            continue
        mixed_input, target = result
        samples.append({
            "instruction": "请将以下西夏文翻译为现代中文：",
            "input": mixed_input,
            "output": target,
            "metadata": {"synthetic": True, "replacement_ratio": round(ratio, 3)},
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Generated {len(samples)} synthetic SFT samples -> {args.output}")


if __name__ == "__main__":
    main()
