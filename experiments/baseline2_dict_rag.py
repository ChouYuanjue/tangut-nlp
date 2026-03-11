"""Baseline 2: Dictionary-Augmented RAG for Tangut translation."""

import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dictionary_utils import BilingualDictionary
from src.prompt_templates import SYSTEM_DICT_RAG, USER_DICT_RAG, build_chat_prompt


def build_glosses_text(tangut_text, dictionary):
    glosses = dictionary.get_glosses(tangut_text)
    lines = []
    for substr, entry in glosses:
        if entry:
            cn = entry.explanationCN if entry.explanationCN else "[未知]"
            en = f" ({entry.explanationEN})" if entry.explanationEN else ""
            gx = f" [拟音: {entry.GX}]" if entry.GX else ""
            lines.append(f"  {substr} = {cn}{en}{gx}")
        else:
            lines.append(f"  {substr} = [未收录]")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", default="data/eval/test_set.jsonl")
    parser.add_argument("--dict-path", default="data/dictionary/dictionary.json")
    parser.add_argument("--model-path", default="models/qwen2.5-7b-instruct")
    parser.add_argument("--output", default="results/baseline2/predictions.jsonl")
    parser.add_argument("--tensor-parallel", type=int, default=2)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams

    dictionary = BilingualDictionary(args.dict_path)
    print(f"Dictionary loaded: {dictionary.get_stats()}")

    with open(args.test_set, "r", encoding="utf-8") as f:
        test_data = [json.loads(line) for line in f]

    prompts = []
    glosses_list = []
    for item in test_data:
        glosses = build_glosses_text(item["input"], dictionary)
        glosses_list.append(glosses)
        user_msg = USER_DICT_RAG.format(tangut_text=item["input"], glosses=glosses)
        prompts.append(build_chat_prompt(SYSTEM_DICT_RAG, user_msg))

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=4096,
    )

    sampling_params = SamplingParams(temperature=0.0, max_tokens=256, top_p=1.0)
    outputs = llm.generate(prompts, sampling_params)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for item, output, glosses in zip(test_data, outputs, glosses_list):
            result = {
                "input": item["input"],
                "reference": item["output"],
                "prediction": output.outputs[0].text.strip(),
                "glosses": glosses,
                "method": "baseline2_dict_rag",
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Baseline 2 complete: {len(outputs)} predictions -> {args.output}")


if __name__ == "__main__":
    main()
