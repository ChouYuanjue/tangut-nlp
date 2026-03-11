"""Combine real and synthetic SFT data with upsampling."""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real", default="data/sft/babelstone_sft.jsonl")
    parser.add_argument("--synthetic", default="data/sft/synthetic_sft.jsonl")
    parser.add_argument("--output", default="data/sft/combined_sft.jsonl")
    parser.add_argument("--upsample-real", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    def load_jsonl(path):
        with open(path, "r", encoding="utf-8") as f:
            return [json.loads(line) for line in f]

    real = load_jsonl(args.real)
    synthetic = load_jsonl(args.synthetic)

    combined = real * args.upsample_real + synthetic
    random.seed(args.seed)
    random.shuffle(combined)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item in combined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Combined: {len(real)} real x{args.upsample_real} + {len(synthetic)} synthetic = {len(combined)} total -> {args.output}")


if __name__ == "__main__":
    main()
