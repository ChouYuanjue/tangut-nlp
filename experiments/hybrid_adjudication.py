"""Hybrid adjudication between frontier and local Tangut title candidates.

This script does not use the gold reference during generation. It only sees:
- Tangut source
- dictionary glosses
- candidate A
- candidate B

It can either:
1. select one candidate verbatim, or
2. synthesize a better final title conservatively.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import AzureOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.azure_openai_config import (
    DEFAULT_AZURE_OPENAI_API_VERSION,
    resolve_azure_openai_config,
)

DEFAULT_API_VERSION = DEFAULT_AZURE_OPENAI_API_VERSION


SYSTEM_PROMPT_BALANCED = """你是一位非常严格的西夏文短标题审校专家。

你的任务是根据西夏文原文、词典释义，以及两个候选中文标题，给出最好的最终标题。

规则：
1. 输出必须是紧凑的中文标题，不得扩写成解释性句子。
2. 不得凭空补充原文没有的内容。
3. 如果某个候选已经明显更好，可以直接原样采用。
4. 如果两个候选各有一部分正确，可以做非常保守的融合或微调，但必须比二者都更像真实标题。
5. 不要机械照抄词典中的泛化佛经名；优先恢复更具体的题名链条。
6. 不要输出分析、解释、标点说明或多行答案。
7. 最终只输出 JSON。"""


SYSTEM_PROMPT_ANTI_HALLUCINATION = """你是一位非常严格的西夏文短标题审校专家。

你的任务是根据西夏文原文、词典释义，以及两个候选中文标题，给出最好的最终标题。

裁决原则按优先级排序如下：
1. 忠于原文和释义，严禁无根据增补。
2. 如果一个候选较流畅、较像常见佛典题名，但明显加入了释义里没有的专名、寺名、地名、泛化佛经名、数字或结构词，必须重罚。
3. 如果一个候选过短、截断，只保留了原文的一小部分，也必须重罚。
4. 在“流畅但可能幻觉”与“略生硬但更保留原文成分”之间，优先后者。
5. 只有在两个候选都同样忠实时，才优先更自然的标题体。
6. 最终输出必须是紧凑标题，不得扩写成解释性句子。
7. 最终只输出 JSON。"""


USER_TEMPLATE = """请完成候选裁决。

【西夏文原文】
{tangut_text}

【逐段词典释义】
{glosses}

【候选A】
{candidate_a}

【候选B】
{candidate_b}

【模式】
{mode_instruction}

请先在内部判断：
- 哪个候选更完整保留原文关键信息？
- 哪个候选有无无根据增补、泛化正规化、过短截断？

请严格输出 JSON，格式如下：
{{"final_title": "...", "basis": "A|B|hybrid", "candidate_a_score": 1-5, "candidate_b_score": 1-5, "reasoning": "一句话理由"}}
"""


MODE_TO_INSTRUCTION = {
    "select": "只能在候选A和候选B中二选一，必须原样复制其中一个，不得改写。",
    "synthesize": "你可以选择A、选择B，或在两者基础上做极小幅度的保守改写，输出更好的最终标题。",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_json_text(text: str) -> dict[str, str]:
    text = (text or "").strip()
    if text.startswith("```json"):
        text = text.replace("```json\n", "", 1)
    elif text.startswith("```"):
        text = text.replace("```\n", "", 1)
    if text.endswith("```"):
        text = text.rpartition("```")[0]
    start_idx = text.find("{")
    end_idx = text.rfind("}") + 1
    if start_idx != -1 and end_idx != 0:
        text = text[start_idx:end_idx]
    parsed = json.loads(text)
    final_title = str(parsed.get("final_title", "")).strip()
    basis = str(parsed.get("basis", "hybrid")).strip()
    reasoning = str(parsed.get("reasoning", "")).strip()
    return {
        "final_title": final_title,
        "basis": basis,
        "candidate_a_score": str(parsed.get("candidate_a_score", "")),
        "candidate_b_score": str(parsed.get("candidate_b_score", "")),
        "reasoning": reasoning,
    }


def load_completed_inputs(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    return {row["input"] for row in load_jsonl(output_path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-a", default="results/frontier_deepseek_v32_fewshot_cot/predictions.jsonl")
    parser.add_argument("--pred-b", default="results/final_gap04_multitask_sigmoid/predictions.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["select", "synthesize"], default="synthesize")
    parser.add_argument("--deployment", default=os.environ.get("AZURE_OPENAI_DEPLOYMENT"))
    parser.add_argument("--endpoint", default=os.environ.get("AZURE_OPENAI_ENDPOINT"))
    parser.add_argument("--api-version", default=DEFAULT_API_VERSION)
    parser.add_argument("--api-key", default=os.environ.get("AZURE_OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=200)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--method-name", default=None)
    parser.add_argument(
        "--profile",
        choices=["balanced", "anti_hallucination"],
        default="balanced",
    )
    parser.add_argument(
        "--copy-if-equal",
        action="store_true",
        help="If the two candidates are identical, copy them without an API call.",
    )
    args = parser.parse_args()

    azure_config = resolve_azure_openai_config(
        api_key=args.api_key,
        endpoint=args.endpoint,
        deployment=args.deployment,
    )
    resolved_deployment = azure_config["deployment"]

    client = AzureOpenAI(
        azure_endpoint=azure_config["endpoint"],
        api_key=azure_config["api_key"],
        api_version=args.api_version,
        timeout=args.timeout,
    )

    rows_a = load_jsonl(Path(args.pred_a))
    rows_b = load_jsonl(Path(args.pred_b))
    if len(rows_a) != len(rows_b):
        raise ValueError(f"Prediction length mismatch: {len(rows_a)} vs {len(rows_b)}")

    paired_rows = list(zip(rows_a, rows_b))
    if args.start_index:
        paired_rows = paired_rows[args.start_index :]
    if args.limit is not None:
        paired_rows = paired_rows[: args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_inputs(output_path)

    system_prompt = {
        "balanced": SYSTEM_PROMPT_BALANCED,
        "anti_hallucination": SYSTEM_PROMPT_ANTI_HALLUCINATION,
    }[args.profile]

    method_name = args.method_name or f"hybrid_{args.mode}_frontier_local_gpt54"
    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as f:
        for idx, (row_a, row_b) in enumerate(paired_rows, start=1):
            if row_a["input"] != row_b["input"]:
                raise ValueError(f"Input mismatch at row {idx}")
            if row_a["input"] in completed:
                print(f"[skip] {idx}: already completed")
                continue

            tangut_input = row_a["input"]
            candidate_a = row_a["prediction"]
            candidate_b = row_b["prediction"]
            glosses = row_a.get("glosses", "")
            if not glosses:
                glosses = row_b.get("glosses", "")

            if args.copy_if_equal and candidate_a == candidate_b:
                record = {
                    "input": tangut_input,
                    "reference": row_a.get("reference", ""),
                    "prediction": candidate_a,
                    "candidate_a": candidate_a,
                    "candidate_b": candidate_b,
                    "glosses": glosses,
                    "basis": "same",
                    "candidate_a_score": "",
                    "candidate_b_score": "",
                    "reasoning": "Candidates identical; copied directly.",
                    "method": method_name,
                    "raw_response": "",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[copy] {idx}/{len(paired_rows)} -> {candidate_a}")
                time.sleep(args.sleep_seconds)
                continue

            user_prompt = USER_TEMPLATE.format(
                tangut_text=tangut_input,
                glosses=glosses or "[无可用词典释义]",
                candidate_a=candidate_a or "[空]",
                candidate_b=candidate_b or "[空]",
                mode_instruction=MODE_TO_INSTRUCTION[args.mode],
            )

            last_error = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    response = client.responses.create(
                        model=resolved_deployment,
                        instructions=system_prompt,
                        input=user_prompt,
                        temperature=args.temperature,
                        max_output_tokens=args.max_output_tokens,
                    )
                    raw_text = response.output_text or "{}"
                    parsed = normalize_json_text(raw_text)
                    record = {
                        "input": tangut_input,
                        "reference": row_a.get("reference", ""),
                        "prediction": parsed["final_title"],
                        "candidate_a": candidate_a,
                        "candidate_b": candidate_b,
                        "glosses": glosses,
                        "basis": parsed["basis"],
                        "candidate_a_score": parsed["candidate_a_score"],
                        "candidate_b_score": parsed["candidate_b_score"],
                        "reasoning": parsed["reasoning"],
                        "method": method_name,
                        "raw_response": raw_text,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"[ok] {idx}/{len(paired_rows)} -> {parsed['final_title']}")
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    print(f"[retry {attempt}/{args.max_retries}] {idx}: {last_error}")
                    if attempt == args.max_retries:
                        record = {
                            "input": tangut_input,
                            "reference": row_a.get("reference", ""),
                            "prediction": "",
                            "candidate_a": candidate_a,
                            "candidate_b": candidate_b,
                            "glosses": glosses,
                            "basis": "error",
                            "candidate_a_score": "",
                            "candidate_b_score": "",
                            "reasoning": last_error,
                            "method": method_name,
                            "raw_response": "",
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                    else:
                        time.sleep(min(8, attempt * 2))

            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
