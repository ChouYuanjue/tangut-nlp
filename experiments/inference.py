"""Unified inference script for any model checkpoint."""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.prompt_templates import (
    SYSTEM_SFT,
    build_chat_prompt,
    build_user_translate,
)


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

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            fix_mistral_regex=True,
        )
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


def build_prompts(test_data, source_label, use_instruction_field, system_prompt):
    prompts = []
    for item in test_data:
        if use_instruction_field and item.get("instruction"):
            user_msg = f"{item['instruction']}\n{item['input']}"
        else:
            user_msg = build_user_translate(item["input"], source_label=source_label)
        prompts.append(build_chat_prompt(system_prompt, user_msg))
    return prompts


def run_vllm_inference(model_path, prompts, tensor_parallel, max_new_tokens):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        top_p=1.0,
    )
    outputs = llm.generate(prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]


def run_hf_inference(model_path, prompts, batch_size, max_new_tokens):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    predictions = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start : start + batch_size]
        batch = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
        )
        batch = {key: value.to(model.device) for key, value in batch.items()}
        input_lengths = batch["attention_mask"].sum(dim=1).tolist()

        with torch.inference_mode():
            generated = model.generate(
                **batch,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        for seq, prompt_len in zip(generated, input_lengths):
            completion = seq[int(prompt_len) :]
            predictions.append(
                tokenizer.decode(completion, skip_special_tokens=True).strip()
            )

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-set", default="data/eval/test_set.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensor-parallel", type=int, default=2)
    parser.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--method-name", default="custom")
    parser.add_argument("--source-label", default="西夏文")
    parser.add_argument(
        "--use-instruction-field",
        action="store_true",
        help="Use each row's instruction field instead of building a fixed user prompt.",
    )
    parser.add_argument(
        "--system-prompt",
        default=SYSTEM_SFT,
        help="System prompt used for inference chat formatting.",
    )
    args = parser.parse_args()

    model_path = merge_lora_if_needed(args.model)

    with open(args.test_set, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]
    prompts = build_prompts(
        test_data,
        source_label=args.source_label,
        use_instruction_field=args.use_instruction_field,
        system_prompt=args.system_prompt,
    )

    if args.backend == "vllm":
        predictions = run_vllm_inference(
            model_path=model_path,
            prompts=prompts,
            tensor_parallel=args.tensor_parallel,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        predictions = run_hf_inference(
            model_path=model_path,
            prompts=prompts,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item, prediction in zip(test_data, predictions):
            result = {
                "input": item["input"],
                "reference": item["output"],
                "prediction": prediction,
                "method": args.method_name,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(
        f"Inference complete ({args.backend}): {len(predictions)} predictions -> {args.output}"
    )


if __name__ == "__main__":
    main()
