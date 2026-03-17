"""Unified inference script for any model checkpoint."""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.prompt_templates import SYSTEM_SFT, USER_TRANSLATE, build_chat_prompt


def merge_lora_if_needed(model_path):
    adapter_config = Path(model_path) / "adapter_config.json"
    if adapter_config.exists():
        print(f"Detected LoRA adapter at {model_path}, merging...")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        with open(adapter_config) as f:
            config = json.load(f)
        base_model_path = config.get("base_model_name_or_path", "models/qwen2.5-7b-instruct")

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path, dtype=torch.bfloat16, trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, model_path)
        merged = model.merge_and_unload()

        merged_path = str(Path(model_path).parent / "merged")
        merged.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        print(f"Merged model saved to {merged_path}")
        return merged_path
    return model_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-set", default="data/eval/test_set.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensor-parallel", type=int, default=2)
    parser.add_argument("--method-name", default="custom")
    args = parser.parse_args()

    model_path = merge_lora_if_needed(args.model)

    from vllm import LLM, SamplingParams

    with open(args.test_set, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=256, top_p=1.0)

    prompts = []
    for item in test_data:
        user_msg = USER_TRANSLATE.format(tangut_text=item["input"])
        prompts.append(build_chat_prompt(SYSTEM_SFT, user_msg))

    outputs = llm.generate(prompts, sampling_params)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item, output in zip(test_data, outputs):
            result = {
                "input": item["input"],
                "reference": item["output"],
                "prediction": output.outputs[0].text.strip(),
                "method": args.method_name,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Inference complete: {len(outputs)} predictions -> {args.output}")


if __name__ == "__main__":
    main()
