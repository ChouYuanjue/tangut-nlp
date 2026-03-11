"""Split tangut_output.jsonl into train/dev/test sets."""

import argparse
import json
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/tangut_output.jsonl")
    parser.add_argument("--test-out", default="data/eval/test_set.jsonl")
    parser.add_argument("--dev-out", default="data/eval/dev_set.jsonl")
    parser.add_argument("--train-out", default="data/sft/babelstone_sft.jsonl")
    parser.add_argument("--test-size", type=int, default=50)
    parser.add_argument("--dev-size", type=int, default=41)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    random.seed(args.seed)
    random.shuffle(data)

    test = data[:args.test_size]
    dev = data[args.test_size:args.test_size + args.dev_size]
    train = data[args.test_size + args.dev_size:]

    for path, split in [(args.test_out, test), (args.dev_out, dev), (args.train_out, train)]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for item in split:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Wrote {len(split)} samples to {path}")


if __name__ == "__main__":
    main()
