#!/usr/bin/env python3
"""Run a reproducible local reference-aware judge with Qwen2.5-7B-Instruct.

This is a fallback evaluator for cases where the Azure-backed judge is
unavailable. It mirrors the JSON structure of ``scripts/run_reference_eval_suite.py``
so downstream summary scripts can be reused unchanged.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import OrderedDict
from pathlib import Path

from vllm import LLM, SamplingParams

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.prompt_templates import build_chat_prompt


DEFAULT_METHODS = [
    "baseline2",
    "baseline3_1_unk",
    "baseline3_2_multitask",
    "final_v2",
    "final_gap02_multitask_sigmoid",
    "final_gap02_multitask_robustwpo",
    "human_reference",
]

SYSTEM_PROMPT = (
    "你是一位严格的西夏文短标题翻译评审员。"
    "你会同时查看西夏文原文、参考译文和候选译文，保守打分。"
    "对于把简短书名扩写成解释性句子、混入拉丁字母或明显杂质、"
    "以及与参考标题不一致的候选，请显著扣分。"
)

USER_TEMPLATE = """请根据下面信息，对候选译文进行严格评分。

【西夏文原文】
{tangut_input}

【参考译文】
{reference}

【候选译文】
{candidate}

评分维度（全部给 1-5 的整数）：
1. reference_agreement
5 = 与参考标题基本一致，只有极小差异；
3 = 有部分重合，但存在明显替换、遗漏或增补；
1 = 与参考明显不符，或严重扩写。

2. source_faithfulness
5 = 对原文关键信息忠实；
3 = 大体可猜，但有不确定或局部错译；
1 = 明显幻觉、错译或偏离原文。

3. title_style_fitness
5 = 保持紧凑书名/标题体；
3 = 略有扩写，但仍像标题；
1 = 变成说明句、夹杂噪声或风格明显不对。

4. overall
综合以上三项给出总评。

请严格只输出一行 JSON，不要输出 markdown，不要解释格式。
格式：
{{"reference_agreement": 4, "source_faithfulness": 4, "title_style_fitness": 5, "overall": 4, "reasoning": "简短理由"}}
"""


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def prediction_path(results_dir: Path, method: str) -> Path:
    custom = {
        "baseline3_2_multitask": results_dir / method / "predictions_cleaned.jsonl",
    }
    return custom.get(method, results_dir / method / "predictions.jsonl")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_prompt(item: dict) -> str:
    user = USER_TEMPLATE.format(
        tangut_input=item["input"],
        reference=item["reference"],
        candidate=item["prediction"],
    )
    return build_chat_prompt(SYSTEM_PROMPT, user)


def clean_json_text(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```json"):
        text = text.replace("```json\n", "", 1)
    elif text.startswith("```"):
        text = text.replace("```\n", "", 1)
    if text.endswith("```"):
        text = text.rpartition("```")[0]
    start_idx = text.find("{")
    end_idx = text.rfind("}") + 1
    if start_idx != -1 and end_idx > start_idx:
        return text[start_idx:end_idx]
    return text


def clamp_score(value: object) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 3
    return max(1, min(5, score))


def parse_output(text: str) -> dict:
    raw_text = text.strip()
    cleaned = clean_json_text(raw_text)
    try:
        parsed = json.loads(cleaned)
        return {
            "reference_agreement": clamp_score(parsed.get("reference_agreement")),
            "source_faithfulness": clamp_score(parsed.get("source_faithfulness")),
            "title_style_fitness": clamp_score(parsed.get("title_style_fitness")),
            "overall": clamp_score(parsed.get("overall")),
            "reasoning": str(parsed.get("reasoning", ""))[:200],
            "judge_raw": raw_text,
        }
    except Exception:
        fallback = {}
        for key in [
            "reference_agreement",
            "source_faithfulness",
            "title_style_fitness",
            "overall",
        ]:
            match = re.search(rf'"?{key}"?\s*[:=]\s*([1-5])', raw_text)
            fallback[key] = clamp_score(match.group(1) if match else None)
        return {
            **fallback,
            "reasoning": "Parse fallback",
            "judge_raw": raw_text,
        }


def summarize(scores: list[dict]) -> dict:
    if not scores:
        return {
            "mean_reference_agreement": 0.0,
            "mean_source_faithfulness": 0.0,
            "mean_title_style_fitness": 0.0,
            "mean_overall": 0.0,
            "scores": [],
        }

    return {
        "mean_reference_agreement": sum(x["reference_agreement"] for x in scores) / len(scores),
        "mean_source_faithfulness": sum(x["source_faithfulness"] for x in scores) / len(scores),
        "mean_title_style_fitness": sum(x["title_style_fitness"] for x in scores) / len(scores),
        "mean_overall": sum(x["overall"] for x in scores) / len(scores),
        "scores": scores,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local Qwen-based reference-aware evaluation suite.")
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS)
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--test-set", default="data/eval/test_set.jsonl")
    parser.add_argument("--output-dir", default="results/reference_eval_local_qwen7b")
    parser.add_argument("--model-path", default="models/qwen2.5-7b-instruct")
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=220)
    parser.add_argument("--limit", type=int, default=None, help="Optional debug limit per method.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    test_set = load_jsonl(Path(args.test_set))
    if args.limit is not None:
        test_set = test_set[: args.limit]

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel,
        dtype="bfloat16",
        trust_remote_code=True,
        max_model_len=2048,
    )
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for method in args.methods:
        pred_path = prediction_path(results_dir, method)
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions for {method}: {pred_path}")

        predictions = load_jsonl(pred_path)
        if args.limit is not None:
            predictions = predictions[: args.limit]

        if len(predictions) != len(test_set):
            raise ValueError(
                f"Prediction/reference length mismatch for {method}: "
                f"{len(predictions)} vs {len(test_set)}"
            )

        items = []
        for pred, ref in zip(predictions, test_set):
            items.append(
                {
                    "input": pred["input"],
                    "reference": ref["output"],
                    "prediction": pred["prediction"],
                    "method": method,
                }
            )

        prompts = [build_prompt(item) for item in items]
        print(f"\n=== Running local reference-aware judge: {method} ({len(prompts)} examples) ===")
        outputs = llm.generate(prompts, sampling_params)

        scored_items = []
        for item, output in zip(items, outputs):
            parsed = parse_output(output.outputs[0].text)
            scored = dict(item)
            scored.update(parsed)
            scored_items.append(scored)

        result = summarize(scored_items)
        out_path = output_dir / f"{method}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        summary_rows.append(
            OrderedDict(
                [
                    ("method", method),
                    ("num_examples", len(result["scores"])),
                    ("mean_reference_agreement", round(result["mean_reference_agreement"], 4)),
                    ("mean_source_faithfulness", round(result["mean_source_faithfulness"], 4)),
                    ("mean_title_style_fitness", round(result["mean_title_style_fitness"], 4)),
                    ("mean_overall", round(result["mean_overall"], 4)),
                    ("output_json", str(out_path)),
                ]
            )
        )
        print(summary_rows[-1])

    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    summary_json.write_text(json.dumps(summary_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_csv(summary_csv, summary_rows)

    print("\n=== Local reference-aware suite summary ===")
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))
    print(f"Wrote {summary_json}")
    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
