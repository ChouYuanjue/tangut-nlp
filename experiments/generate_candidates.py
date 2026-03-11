"""Generate N=5 candidates per input, score with reward function, build DPO pairs."""

import sys
import argparse
import json
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.prompt_templates import SYSTEM_SFT, build_chat_prompt

N_CANDIDATES = 5
ALPHA = 1.0
BETA = 0.01


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-model", default="checkpoints/sft/merged")
    parser.add_argument("--train-data", default="data/sft/combined_sft.jsonl")
    parser.add_argument("--reward-dict", default="data/dictionary/reward_dict.json")
    parser.add_argument("--ppl-model", default="models/qwen2.5-0.5b")
    parser.add_argument("--output", default="data/dpo/dpo_pairs.jsonl")
    parser.add_argument("--max-inputs", type=int, default=5000)
    parser.add_argument("--tensor-parallel", type=int, default=2)
    args = parser.parse_args()

    from vllm import LLM, SamplingParams
    from eval.lexical_coverage import LexicalCoverageScorer
    from eval.perplexity import PerplexityScorer

    with open(args.train_data, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    data = data[:args.max_inputs]

    llm = LLM(
        model=args.sft_model,
        tensor_parallel_size=args.tensor_parallel,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
    )

    sampling_params = SamplingParams(
        temperature=0.8, max_tokens=256, top_p=0.95, n=N_CANDIDATES,
    )

    prompts = []
    for item in data:
        user = f"{item['instruction']}\n{item['input']}"
        prompts.append(build_chat_prompt(SYSTEM_SFT, user))

    print(f"Generating {N_CANDIDATES} candidates for {len(prompts)} inputs...")
    outputs = llm.generate(prompts, sampling_params)

    del llm
    import torch
    torch.cuda.empty_cache()

    print("Scoring candidates...")
    lex_scorer = LexicalCoverageScorer(args.reward_dict)
    ppl_scorer = PerplexityScorer(model_path=args.ppl_model, device="cuda:0")

    dpo_pairs = []
    for item, output in zip(data, outputs):
        candidates = [o.text.strip() for o in output.outputs]
        scored = []
        for cand in candidates:
            if not cand:
                continue
            lex = lex_scorer.score(item["input"], cand)
            ppl = ppl_scorer.score(cand)
            log_ppl = math.log(ppl + 1e-8)
            reward = ALPHA * lex - BETA * log_ppl
            scored.append({"text": cand, "lex": lex, "ppl": ppl, "reward": reward})
        if len(scored) < 2:
            continue
        scored.sort(key=lambda x: x["reward"], reverse=True)
        if scored[0]["reward"] - scored[-1]["reward"] < 0.05:
            continue
        dpo_pairs.append({
            "prompt": f"{item['instruction']}\n{item['input']}",
            "chosen": scored[0]["text"],
            "rejected": scored[-1]["text"],
            "chosen_reward": scored[0]["reward"],
            "rejected_reward": scored[-1]["reward"],
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    print(f"Generated {len(dpo_pairs)} DPO preference pairs -> {args.output}")


if __name__ == "__main__":
    main()
