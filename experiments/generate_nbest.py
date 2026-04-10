"""Generate n-best candidates for Tangut title translation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.prompt_templates import SYSTEM_SFT, USER_TRANSLATE, build_chat_prompt


def merge_lora_if_needed(model_path: str) -> str:
    adapter_config = Path(model_path) / "adapter_config.json"
    merged_path = Path(model_path).parent / "merged"
    if adapter_config.exists() and merged_path.exists():
        return str(merged_path)
    if not adapter_config.exists():
        return model_path

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    with open(adapter_config, "r", encoding="utf-8") as f:
        config = json.load(f)
    base_model_path = config.get("base_model_name_or_path", "models/qwen2.5-7b-instruct")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base_model, model_path)
    merged = model.merge_and_unload()

    merged_path.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))
    return str(merged_path)


def build_prompts(test_rows: list[dict]) -> list[str]:
    prompts = []
    for item in test_rows:
        user_msg = USER_TRANSLATE.format(tangut_text=item["input"])
        prompts.append(build_chat_prompt(SYSTEM_SFT, user_msg))
    return prompts


def run_vllm_nbest(
    model_path: str,
    prompts: list[str],
    tensor_parallel: int,
    num_candidates: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    repetition_penalty: float,
) -> list[list[str]]:
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
    )

    sampling_params = SamplingParams(
        n=num_candidates,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        repetition_penalty=repetition_penalty,
    )
    outputs = llm.generate(prompts, sampling_params)
    results: list[list[str]] = []
    for output in outputs:
        results.append([item.text.strip() for item in output.outputs])
    return results


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    kept = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        kept.append(item)
    return kept


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test-set", default="data/eval/test_set.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    args = parser.parse_args()

    model_path = merge_lora_if_needed(args.model)
    with open(args.test_set, "r", encoding="utf-8") as f:
        test_rows = [json.loads(line) for line in f if line.strip()]
    prompts = build_prompts(test_rows)

    candidate_lists = run_vllm_nbest(
        model_path=model_path,
        prompts=prompts,
        tensor_parallel=args.tensor_parallel,
        num_candidates=args.num_candidates,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row, candidates in zip(test_rows, candidate_lists):
            record = {
                "input": row["input"],
                "reference": row.get("output", ""),
                "candidates": dedupe_keep_order(candidates),
                "method": "local_nbest",
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(candidate_lists)} n-best rows -> {output_path}")


if __name__ == "__main__":
    main()
