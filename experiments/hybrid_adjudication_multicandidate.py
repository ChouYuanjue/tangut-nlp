"""Listwise hybrid adjudication over multiple Tangut title candidates.

The adjudicator never sees the gold reference at generation time. It only uses:
- Tangut source
- dictionary glosses
- multiple candidate Chinese titles from different systems

It can either:
1. select one candidate verbatim, or
2. synthesize a minimally repaired title from the candidate pool.
"""

from __future__ import annotations

import argparse
import json
import os
import string
import sys
import time
from pathlib import Path

from openai import AzureOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.azure_openai_config import (
    DEFAULT_AZURE_OPENAI_API_VERSION,
    resolve_azure_openai_config,
)

DEFAULT_API_VERSION = DEFAULT_AZURE_OPENAI_API_VERSION

SYSTEM_PROMPT_HISTORICAL_TITLE = """你是一位极其严格的西夏文历史短标题审校专家。

你的任务是根据西夏文原文、逐段词典释义，以及多个候选中文标题，给出最可信的最终中文题名。

裁决原则按优先级排序如下：
1. 这是“历史短标题/书名”任务，不是释义任务。输出必须保持紧凑标题体，不得扩写成解释性句子。
2. 词典释义只是局部释读线索，不等于规范汉文题名。不要机械照抄释义做百科式扩写。
3. 当“较流畅、较像常见佛典名”的候选和“较保守、较贴近原文字面或成分”的候选冲突时，优先后者。
4. 对极短原文，长篇专名扩写风险极高；如果一个候选明显过长，而另一个更短、更保守且仍保留关键信息，优先更短者。
5. 可以接受常见题名化词汇替换，例如“部/經/論/序/記/頌/門/根”，前提是没有凭空新增实体、地名、寺名、数字或结构。
6. 只有在多个候选互补且证据充分时，才允许极小幅度融合或修复；禁止引入所有候选和释义里都没有的新信息。
7. 如果拿不准，优先选择更短、更保守、增补更少的标题。
8. 只输出严格 JSON。"""

SYSTEM_PROMPT_ANTI_EXPANSION = """你是一位极其严格的西夏文历史短标题审校专家。

你的核心任务不是把释义翻成流畅句子，而是在多个候选里找出最不胡编、最像真实题名的版本。

强制规则：
1. 严禁因为词典释义较长，就把极短原文扩写成寺名、地名、完整佛典名或解释性短语。
2. 词典释义可能是解释、训释或后设说明，不一定是规范题名；不要对释义作机械直译。
3. 如果一个候选更像“保守的标题复原”，另一个更像“顺手正规化后的熟悉佛经名”，优先前者。
4. 如果一个候选只比另一个更流畅，却多出了无证据的实体、类别词、结构词或篇幅，必须重罚。
5. 若允许改写，也只能做外科手术式微调：只可重组候选中已有成分，或使用释义里直接对应的单个题名成分；不得发明新的信息。
6. 最终输出必须是紧凑中文标题，不得附加解释，不得输出多行。
7. 只输出严格 JSON。"""

SYSTEM_PROMPT_CATALOG_LITERALIST = """你是一位极其严格的西夏文目录短标题审校专家。

这类题名常常并不流畅，也可能保留半音译、半意译、半训释的历史链条。你的目标不是把它们改写成熟悉现代汉语，而是尽量保住题名链条。

强制规则：
1. 优先保留不透明但稳定的题名链条，不要把它们顺手正规化成更熟悉、但证据更弱的佛典名。
2. 若多个候选共享同一保守片段（如共同前缀、共同后缀或共同核心词），该片段往往比某个单独候选的流畅改写更可信。
3. “白母、正理滴、金剛王、順要論、五部經”这类略生涩或偏目录体的形式，不应因不够现代流畅而自动判错。
4. 对极短原文，必须重罚长篇扩写；对较长原文，必须重罚只顾流畅而丢失链条成分的正规化标题。
5. 若允许修复，只能围绕候选之间已经出现的稳定片段做极小改写；不得发明候选池和释义都没有的新信息。
6. 如果拿不准，优先选择更保守、更少解释、更少现代化正规化的标题。
7. 最终输出必须是紧凑中文标题，且只输出严格 JSON。"""

USER_TEMPLATE = """请完成候选裁决。

【西夏文原文】
{tangut_text}

【长度提示】
- 原文字符数：{source_length}

【逐段词典释义】
{glosses}

【候选标题】
{candidate_block}

【候选共识提示】
{consensus_block}

【模式】
{mode_instruction}

请先在内部判断：
- 哪个候选更像“历史短标题”而不是释义扩写？
- 哪个候选更少无根据正规化、无根据增补、过短截断或过长扩写？
- 如果需要融合，是否只靠候选池里现成成分就能做出更保守的修复？

请严格输出 JSON，格式如下：
{{"final_title": "...", "basis": "A|B|C|hybrid", "selected_ids": ["A"], "scores": {{"A": 4, "B": 2, "C": 3}}, "reasoning": "一句话理由"}}
"""

MODE_TO_INSTRUCTION = {
    "select": "只能在候选中选择一个，必须原样复制其中一个，不得改写。",
    "synthesize": "你可以选择某个候选原样输出，或基于多个候选做极小幅度的保守修复，但不得引入候选池和释义之外的新内容。",
}


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_pred_spec(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"Expected NAME=PATH for --pred, got: {spec}")
    name, path = spec.split("=", 1)
    return name.strip(), Path(path).expanduser()


def normalize_json_text(text: str) -> dict:
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

    selected_ids = parsed.get("selected_ids", [])
    if isinstance(selected_ids, str):
        selected_ids = [selected_ids] if selected_ids.strip() else []

    scores = parsed.get("scores", {})
    normalized_scores = {}
    if isinstance(scores, dict):
        for key, value in scores.items():
            try:
                normalized_scores[str(key)] = max(1, min(5, int(value)))
            except Exception:
                continue

    return {
        "final_title": str(parsed.get("final_title", "")).strip(),
        "basis": str(parsed.get("basis", "hybrid")).strip(),
        "selected_ids": [str(x).strip() for x in selected_ids if str(x).strip()],
        "scores": normalized_scores,
        "reasoning": str(parsed.get("reasoning", "")).strip(),
    }


def load_completed_inputs(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    return {row["input"] for row in load_jsonl(output_path)}


def build_candidate_entries(
    rows_by_name: dict[str, list[dict]],
    row_idx: int,
) -> tuple[list[dict], str, str, str]:
    letters = string.ascii_uppercase
    seen_predictions: set[str] = set()
    candidates = []
    reference = ""
    tangut_input = ""
    glosses = ""

    for name, rows in rows_by_name.items():
        row = rows[row_idx]
        if not tangut_input:
            tangut_input = row["input"]
        elif tangut_input != row["input"]:
            raise ValueError(f"Input mismatch at row {row_idx + 1}: {name}")

        if not reference:
            reference = row.get("reference", "")
        if not glosses:
            glosses = row.get("glosses", "")

        prediction = (row.get("prediction", "") or "").strip()
        if not prediction or prediction in seen_predictions:
            continue
        seen_predictions.add(prediction)
        label = letters[len(candidates)]
        candidates.append(
            {
                "label": label,
                "source": name,
                "prediction": prediction,
            }
        )

    return candidates, tangut_input, reference, glosses


def build_candidate_block(candidates: list[dict]) -> str:
    lines = []
    for candidate in candidates:
        lines.append(
            f"[{candidate['label']}] ({candidate['source']}, {len(candidate['prediction'])}字)\n"
            f"{candidate['prediction']}"
        )
    return "\n\n".join(lines)


def common_prefix(a: str, b: str) -> str:
    out = []
    for ch_a, ch_b in zip(a, b):
        if ch_a != ch_b:
            break
        out.append(ch_a)
    return "".join(out)


def common_suffix(a: str, b: str) -> str:
    out = []
    for ch_a, ch_b in zip(reversed(a), reversed(b)):
        if ch_a != ch_b:
            break
        out.append(ch_a)
    return "".join(reversed(out))


def build_consensus_block(candidates: list[dict]) -> str:
    hints = []
    for i, left in enumerate(candidates):
        for right in candidates[i + 1 :]:
            prefix = common_prefix(left["prediction"], right["prediction"])
            suffix = common_suffix(left["prediction"], right["prediction"])
            pair = f"{left['label']}/{right['label']}"
            if len(prefix) >= 2:
                hints.append(f"- {pair} 共享前缀：{prefix}")
            if len(suffix) >= 2 and suffix != prefix:
                hints.append(f"- {pair} 共享后缀：{suffix}")
    return "\n".join(hints) if hints else "[无明显候选共识片段]"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        help="Repeated NAME=PATH spec for candidate prediction files.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["select", "synthesize"], default="synthesize")
    parser.add_argument(
        "--profile",
        choices=["historical_title", "anti_expansion", "catalog_literalist"],
        default="historical_title",
    )
    parser.add_argument("--deployment", default=os.environ.get("AZURE_OPENAI_DEPLOYMENT"))
    parser.add_argument("--endpoint", default=os.environ.get("AZURE_OPENAI_ENDPOINT"))
    parser.add_argument("--api-version", default=DEFAULT_API_VERSION)
    parser.add_argument("--api-key", default=os.environ.get("AZURE_OPENAI_API_KEY"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=260)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--method-name", default=None)
    parser.add_argument(
        "--copy-if-identical",
        action="store_true",
        help="If all candidate predictions collapse to one unique string, copy it directly.",
    )
    args = parser.parse_args()

    pred_specs = [parse_pred_spec(spec) for spec in args.pred]
    if len(pred_specs) < 2:
        raise ValueError("Provide at least two --pred specs.")

    rows_by_name = {name: load_jsonl(path) for name, path in pred_specs}
    lengths = {name: len(rows) for name, rows in rows_by_name.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Prediction length mismatch: {lengths}")

    total_rows = next(iter(lengths.values()))
    indices = list(range(total_rows))
    if args.start_index:
        indices = indices[args.start_index :]
    if args.limit is not None:
        indices = indices[: args.limit]

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

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_inputs(output_path)

    system_prompt = {
        "historical_title": SYSTEM_PROMPT_HISTORICAL_TITLE,
        "anti_expansion": SYSTEM_PROMPT_ANTI_EXPANSION,
        "catalog_literalist": SYSTEM_PROMPT_CATALOG_LITERALIST,
    }[args.profile]
    joined_names = "_".join(name for name, _ in pred_specs)
    method_name = args.method_name or f"hybrid_{args.mode}_{joined_names}_gpt54"

    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as f:
        for out_idx, row_idx in enumerate(indices, start=1):
            candidates, tangut_input, reference, glosses = build_candidate_entries(rows_by_name, row_idx)
            if tangut_input in completed:
                print(f"[skip] {out_idx}: already completed")
                continue
            if not candidates:
                raise ValueError(f"No usable candidates at row {row_idx + 1}")

            if args.copy_if_identical and len(candidates) == 1:
                record = {
                    "input": tangut_input,
                    "reference": reference,
                    "prediction": candidates[0]["prediction"],
                    "candidates": candidates,
                    "glosses": glosses,
                    "basis": "same",
                    "selected_ids": [candidates[0]["label"]],
                    "scores": {},
                    "reasoning": "All candidate systems produced the same title; copied directly.",
                    "method": method_name,
                    "raw_response": "",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[copy] {out_idx}/{len(indices)} -> {candidates[0]['prediction']}")
                time.sleep(args.sleep_seconds)
                continue

            user_prompt = USER_TEMPLATE.format(
                tangut_text=tangut_input,
                source_length=len(tangut_input),
                glosses=glosses or "[无可用词典释义]",
                candidate_block=build_candidate_block(candidates),
                consensus_block=build_consensus_block(candidates),
                mode_instruction=MODE_TO_INSTRUCTION[args.mode],
            )

            last_error: Exception | None = None
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
                        "reference": reference,
                        "prediction": parsed["final_title"],
                        "candidates": candidates,
                        "glosses": glosses,
                        "basis": parsed["basis"],
                        "selected_ids": parsed["selected_ids"],
                        "scores": parsed["scores"],
                        "reasoning": parsed["reasoning"],
                        "method": method_name,
                        "raw_response": raw_text,
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"[ok] {out_idx}/{len(indices)} -> {parsed['final_title']}")
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt == args.max_retries:
                        break
                    time.sleep(min(2 ** attempt, 8))

            if last_error is not None:
                raise RuntimeError(
                    f"Failed at row {row_idx + 1} after {args.max_retries} attempts: {last_error}"
                ) from last_error

            time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
