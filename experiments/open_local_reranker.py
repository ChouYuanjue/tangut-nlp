#!/usr/bin/env python3
"""Open local candidate reranker with Qwen2.5-7B-Instruct.

This script provides an open alternative to the closed catalog adjudicator.
It does not see the gold reference. It reads an existing candidate pool and
selects one candidate verbatim using a locally hosted instruction model.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path

import sacrebleu
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from experiments.hybrid_adjudication_multicandidate import (  # noqa: E402
    build_candidate_block,
    build_candidate_entries,
    build_consensus_block,
    load_jsonl,
)
from experiments.open_hybrid_heuristic import BAD_PATTERN  # noqa: E402


DIAG_TITLE_SUFFIXES = set("經論記疏頌儀義傳錄贊序字品")

SYSTEM_PROMPT = """你是一位极其严格的西夏文目录短标题审校专家。

你的任务是在多个候选中文标题中选出最可信的最终题名。

强制规则：
1. 这是历史短标题任务，不是释义任务；严禁为了流畅而扩写成说明句。
2. 词典释义只是局部线索，不等于规范汉文题名；不要机械照抄宽泛释义。
3. 优先选择更保守、更少增补、更少无根据正规化的候选。
4. 对极短原文，必须重罚长篇扩写；对较长原文，必须重罚明显截断。
5. 若多个候选共享稳定前缀、后缀或核心片段，这种共识通常比单个流畅改写更可信。
6. 你只能在给定候选中选择一个，必须原样复制，不得改写，不得融合，不得补词。
7. 如果拿不准，优先选择更短、更克制、题名体更强的候选。
8. 最终只能输出一个大写字母（A、B、C...）表示你选择的候选。"""

USER_TEMPLATE = """请从候选标题中选择最可信的一项。

【西夏文原文】
{tangut_text}

【原文长度】
{source_length}

【逐段词典释义】
{glosses}

【候选标题】
{candidate_block}

【候选共识提示】
{consensus_block}

请只输出一个大写字母（例如 A），不要输出 JSON，不要解释，不要多写任何别的内容。
"""


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prompt(
    *,
    tokenizer,
    tangut_text: str,
    glosses: str,
    candidates: list[dict],
) -> str:
    user = USER_TEMPLATE.format(
        tangut_text=tangut_text,
        source_length=len(tangut_text),
        glosses=glosses,
        candidate_block=build_candidate_block(candidates),
        consensus_block=build_consensus_block(candidates),
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


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


def parse_basis(text: str, valid_labels: set[str]) -> tuple[str | None, str]:
    raw = (text or "").strip()
    direct = re.search(r"\b([A-Z])\b", raw)
    if direct:
        basis = direct.group(1).upper()
        if basis in valid_labels:
            return basis, "Letter-only selection"

    cleaned = clean_json_text(raw)
    try:
        parsed = json.loads(cleaned)
        basis = str(parsed.get("basis", "")).strip().upper()
        if basis in valid_labels:
            return basis, "JSON fallback"
    except Exception:
        pass

    match = re.search(r'"?basis"?\s*[:=]\s*"?(?P<label>[A-Z])"?', raw, flags=re.IGNORECASE)
    if match:
        basis = match.group("label").upper()
        if basis in valid_labels:
            return basis, "Regex fallback"
    return None, "Fallback to frontier"


def batch_generate(
    *,
    model,
    tokenizer,
    prompts: list[str],
    batch_size: int,
    max_new_tokens: int,
) -> list[str]:
    outputs = []
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
            outputs.append(tokenizer.decode(completion, skip_special_tokens=True).strip())
    return outputs


def corpus_chrf(hyps: list[str], refs: list[str]) -> float:
    return sacrebleu.corpus_chrf(hyps, [refs], char_order=6, word_order=2, beta=2).score


def evaluate_predictions(rows: list[dict], frontier_name: str) -> OrderedDict:
    refs = [row["reference"] for row in rows]
    hyps = [row["prediction"] for row in rows]
    exact = sum(int(h == r) for h, r in zip(hyps, refs))
    contamination = sum(int(bool(BAD_PATTERN.search(h))) for h in hyps)

    suffix_total = 0
    suffix_ok = 0
    for hyp, ref in zip(hyps, refs):
        ref_suffixes = {ch for ch in ref if ch in DIAG_TITLE_SUFFIXES}
        if not ref_suffixes:
            continue
        suffix_total += 1
        if all(ch in hyp for ch in ref_suffixes):
            suffix_ok += 1

    basis_counts = {}
    switched_from_frontier = 0
    for row in rows:
        basis = row["selector_basis"]
        basis_counts[basis] = basis_counts.get(basis, 0) + 1
        if basis != frontier_name:
            switched_from_frontier += 1

    return OrderedDict(
        [
            ("num_examples", len(rows)),
            ("corpus_chrf", round(corpus_chrf(hyps, refs), 4)),
            ("exact_match", exact),
            ("exact_match_rate", round(exact / len(rows), 4)),
            ("contamination_rate", round(contamination / len(rows), 4)),
            (
                "length_ratio",
                round(sum(len(h) for h in hyps) / max(1, sum(len(r) for r in refs)), 4),
            ),
            ("title_suffix_ok", suffix_ok),
            ("title_suffix_total", suffix_total),
            ("switched_from_frontier", switched_from_frontier),
            ("basis_counts", json.dumps(basis_counts, ensure_ascii=False, sort_keys=True)),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an open local reranker over Tangut title candidates.")
    parser.add_argument(
        "--pred",
        action="append",
        required=True,
        help="Repeated NAME=PATH spec for candidate prediction files.",
    )
    parser.add_argument("--output", required=True, help="Path to save predictions.jsonl.")
    parser.add_argument("--summary", default=None, help="Optional path to save a summary JSON.")
    parser.add_argument("--model-path", default="models/qwen2.5-7b-instruct")
    parser.add_argument("--frontier-name", default="frontier")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows_by_name = {}
    for spec in args.pred:
        if "=" not in spec:
            raise ValueError(f"Expected NAME=PATH, got: {spec}")
        name, raw_path = spec.split("=", 1)
        rows_by_name[name.strip()] = load_jsonl(Path(raw_path).expanduser())

    lengths = {len(v) for v in rows_by_name.values()}
    if len(lengths) != 1:
        raise ValueError(f"Prediction length mismatch: {lengths}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    num_rows = next(iter(lengths))
    indices = list(range(num_rows))
    if args.limit is not None:
        indices = indices[: args.limit]

    prompt_payloads = []
    output_rows = []
    for idx in indices:
        candidates, tangut_input, reference, glosses = build_candidate_entries(rows_by_name, idx)
        if len(candidates) == 1:
            only = candidates[0]
            output_rows.append(
                {
                    "_row_idx": idx,
                    "input": tangut_input,
                    "reference": reference,
                    "prediction": only["prediction"],
                    "candidates": candidates,
                    "glosses": glosses,
                    "basis": only["label"],
                    "selected_ids": [only["label"]],
                    "reasoning": "Only unique candidate.",
                    "method": "open_local_qwen_reranker",
                    "selector_basis": only["source"],
                    "raw_response": "",
                }
            )
            continue

        prompt_payloads.append(
            {
                "idx": idx,
                "tangut_input": tangut_input,
                "reference": reference,
                "glosses": glosses,
                "candidates": candidates,
                "prompt": build_prompt(
                    tokenizer=tokenizer,
                    tangut_text=tangut_input,
                    glosses=glosses,
                    candidates=candidates,
                ),
            }
        )

    prompts = [item["prompt"] for item in prompt_payloads]
    raw_outputs = batch_generate(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    for payload, raw_output in zip(prompt_payloads, raw_outputs):
        valid_labels = {item["label"] for item in payload["candidates"]}
        basis, reasoning = parse_basis(raw_output, valid_labels)
        if basis is None:
            frontier = next(
                (item for item in payload["candidates"] if item["source"] == args.frontier_name),
                payload["candidates"][0],
            )
            chosen = frontier
            reasoning = f"{reasoning}; defaulted to {chosen['label']}."
        else:
            chosen = next(item for item in payload["candidates"] if item["label"] == basis)

        output_rows.append(
            {
                "_row_idx": payload["idx"],
                "input": payload["tangut_input"],
                "reference": payload["reference"],
                "prediction": chosen["prediction"],
                "candidates": payload["candidates"],
                "glosses": payload["glosses"],
                "basis": chosen["label"],
                "selected_ids": [chosen["label"]],
                "reasoning": reasoning,
                "method": "open_local_qwen_reranker",
                "selector_basis": chosen["source"],
                "raw_response": raw_output,
            }
        )

    output_rows.sort(key=lambda row: row["_row_idx"])
    for row in output_rows:
        row.pop("_row_idx", None)
    write_jsonl(Path(args.output), output_rows)
    metrics = evaluate_predictions(output_rows, args.frontier_name)
    if args.summary:
        Path(args.summary).write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
