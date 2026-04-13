"""Baseline 1: Zero-shot Tangut translation via Qwen2.5-7B-Instruct."""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.prompt_templates import (
    SYSTEM_ZEROSHOT,
    build_chat_prompt,
    build_user_translate,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", default="data/eval/test_set.jsonl")
    parser.add_argument("--model-path", default="models/qwen2.5-7b-instruct")
    parser.add_argument("--output", default="results/baseline1/predictions.jsonl")
    parser.add_argument("--tensor-parallel", type=int, default=2)
    parser.add_argument("--source-label", default="西夏文")
    parser.add_argument(
        "--use-instruction-field",
        action="store_true",
        help="Use each row's instruction field instead of building a fixed user prompt.",
    )
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    with open(args.test_set, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=256, top_p=1.0)

    prompts = []
    for item in test_data:
        if args.use_instruction_field and item.get("instruction"):
            user_msg = f"{item['instruction']}\n{item['input']}"
        else:
            user_msg = build_user_translate(item["input"], source_label=args.source_label)
        prompts.append(build_chat_prompt(SYSTEM_ZEROSHOT, user_msg))

    outputs = llm.generate(prompts, sampling_params)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item, output in zip(test_data, outputs):
            result = {
                "input": item["input"],
                "reference": item["output"],
                "prediction": output.outputs[0].text.strip(),
                "method": "baseline1_zeroshot",
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Baseline 1 complete: {len(outputs)} predictions -> {args.output}")


if __name__ == "__main__":
    main()
