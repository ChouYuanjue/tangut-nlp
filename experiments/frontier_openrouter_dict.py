"""Frontier prompt-only Tangut translation via OpenRouter chat completions.

This script is designed to answer a reviewer-style question directly:
"If we swap in a stronger frontier Chinese LLM and push prompt engineering
harder, does the task still require task-specific adaptation?"

It reuses the repository's dictionary gloss construction, but replaces the
local base model with an external chat model served through OpenRouter.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import requests

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dictionary_utils import BilingualDictionary


DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "deepseek/deepseek-v3.2"


SYSTEM_PROMPT = """你是一位非常严格的西夏文书名/短标题翻译专家。

你的任务不是写解释、摘要或现代白话句子，而是把西夏文短文本还原成尽可能准确、紧凑的现代中文标题。

必须同时满足这些要求：
1. 优先输出“标题/书名体”，而不是解释性句子。
2. 只能根据给定西夏文和词典释义做判断；不能虚构上下文。
3. 多义词必须全局消歧，宁可保守，也不要把多个释义拼成冗长句子。
4. 如果某个词条明显只是宽泛义项、错误引导或泛化佛经名，不要机械照抄。
4a. 特别是当某段释义像“《某某经》(佛经)”这种泛化书名，而前文已经形成了更具体的题名链条时，通常只把它当作题名类别信号，如“經”，不要把那个泛化经名整体抄进结果。
5. 对佛典标题，优先保持常见汉语标题格式，如“經、論、記、疏、頌、儀、義、傳、錄、贊、序、品、文、觀、根、次”等结尾；如果证据不足，不要硬补。
6. 不要输出解释、引号、书名号、括号、标点、分点、注释、英文、拼音、JSON 以外的任何文本。
7. 最终输出应尽量短，长度通常接近原始参考标题，而不是扩写成长句。

你可以在内部做分步推理，但最终只输出标题本身，不要输出任何分析。"""


USER_TEMPLATE = """请根据下列信息生成最终中文标题。

【任务】
把西夏文短文本翻译成现代中文标题。

【西夏文原文】
{tangut_text}

【逐段词典释义】
{glosses}

【内部推理要求】
- 先识别哪些词更像专名、音译佛典词、结构词、题名后缀。
- 再做全局消歧，避免把所有字面义累加成解释性句子。
- 如果出现多个候选，优先选择更像真实中文书名、且更短更克制的一项。
- 尤其警惕把泛化词典项误译成固定佛经名、或把动词义扩写成现代说明句。
- 如果最后一段像泛化佛经名，但整句更像“某某陀羅尼經/某某論/某某記”，应优先恢复那个更具体的标题，而不是照抄泛化经名。

【输出格式】
只输出最终标题本身，不得包含任何额外文本。
"""


FEWSHOT_EXAMPLE_IDS = [1, 6, 15, 16]


def build_fewshot_block(dev_rows: list[dict], dictionary: BilingualDictionary) -> str:
    blocks = []
    for shot_id in FEWSHOT_EXAMPLE_IDS:
        item = dev_rows[shot_id]
        glosses = build_glosses_text(item["input"], dictionary)
        blocks.append(
            f"【示例{len(blocks) + 1}】\n"
            f"西夏文：{item['input']}\n"
            f"逐段释义：\n{glosses}\n"
            f"最终标题：{item['output']}"
        )
    return "\n\n".join(blocks)

def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_glosses_text(tangut_text: str, dictionary: BilingualDictionary) -> str:
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


def extract_message_text(raw: dict[str, Any]) -> str:
    message = raw["choices"][0]["message"]
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for item in content:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    chunks.append(str(item["text"]))
                elif "text" in item and item["text"]:
                    chunks.append(str(item["text"]))
        return "\n".join(chunks)
    return ""


def normalize_prediction(text: str) -> str:
    text = text.strip()
    if not text:
        return ""

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    if text.startswith("{") and text.endswith("}"):
        try:
            parsed = json.loads(text)
            candidate = parsed.get("final_title") or parsed.get("title") or parsed.get("answer")
            if candidate:
                text = str(candidate).strip()
        except json.JSONDecodeError:
            pass

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        text = line
        break

    text = re.sub(r"^(最终标题|答案|译文|输出)[:：]\s*", "", text)
    text = text.strip().strip("“”\"'`")
    if text.startswith("《") and text.endswith("》"):
        text = text[1:-1].strip()
    return text


def load_completed_inputs(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    completed = set()
    for row in load_jsonl(output_path):
        completed.add(row["input"])
    return completed


def call_openrouter(
    *,
    endpoint: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout: int,
    enable_reasoning: bool,
) -> dict:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if enable_reasoning:
        payload["reasoning"] = {"enabled": True}

    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", default="data/eval/test_set.jsonl")
    parser.add_argument("--dict-path", default="data/dictionary/dictionary.json")
    parser.add_argument("--dev-set", default="data/eval/dev_set.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--method-name", default="frontier_deepseek_v32_dict_cot")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--sleep-seconds", type=float, default=0.5)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--enable-reasoning", action="store_true")
    parser.add_argument(
        "--prompt-mode",
        choices=["strict", "fewshot"],
        default="strict",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing OpenRouter API key. Use --api-key or OPENROUTER_API_KEY.")

    test_rows = load_jsonl(Path(args.test_set))
    if args.start_index:
        test_rows = test_rows[args.start_index :]
    if args.limit is not None:
        test_rows = test_rows[: args.limit]

    dictionary = BilingualDictionary(args.dict_path)
    fewshot_block = ""
    if args.prompt_mode == "fewshot":
        dev_rows = load_jsonl(Path(args.dev_set))
        fewshot_block = (
            "下面是若干同任务示例。注意这些示例的共同规律：结果保持短标题体，"
            "不会把词典里的泛化佛经名机械照抄成最终题名。\n\n"
            f"{build_fewshot_block(dev_rows, dictionary)}\n\n"
        )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed_inputs = load_completed_inputs(output_path)

    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as f:
        for idx, item in enumerate(test_rows, start=1):
            if item["input"] in completed_inputs:
                print(f"[skip] {idx}: already completed")
                continue

            glosses = build_glosses_text(item["input"], dictionary)
            user_prompt = USER_TEMPLATE.format(
                tangut_text=item["input"],
                glosses=glosses,
            )
            if fewshot_block:
                user_prompt = fewshot_block + user_prompt

            last_error = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    raw = call_openrouter(
                        endpoint=args.endpoint,
                        api_key=api_key,
                        model=args.model,
                        system_prompt=SYSTEM_PROMPT,
                        user_prompt=user_prompt,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        timeout=args.timeout,
                        enable_reasoning=args.enable_reasoning,
                    )
                    content = extract_message_text(raw)
                    final_title = normalize_prediction(content)
                    record = {
                        "input": item["input"],
                        "reference": item["output"],
                        "prediction": final_title,
                        "glosses": glosses,
                        "method": args.method_name,
                        "model": args.model,
                        "prompt_variant": f"{args.prompt_mode}_title_only_cot",
                        "literal_draft": "",
                        "notes": "",
                        "raw_response": content,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"[ok] {idx}/{len(test_rows)} -> {final_title}")
                    break
                except Exception as exc:  # noqa: BLE001
                    last_error = str(exc)
                    print(f"[retry {attempt}/{args.max_retries}] {idx}: {last_error}")
                    if attempt == args.max_retries:
                        record = {
                            "input": item["input"],
                            "reference": item["output"],
                            "prediction": "",
                            "glosses": glosses,
                            "method": args.method_name,
                            "model": args.model,
                            "prompt_variant": f"{args.prompt_mode}_title_only_cot",
                            "literal_draft": "",
                            "notes": f"ERROR: {last_error}",
                            "raw_response": "",
                        }
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        f.flush()
                    else:
                        time.sleep(min(8, attempt * 2))

            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
