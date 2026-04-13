#!/usr/bin/env python3
"""Local Qwen decoding under the frontier DeepSeek prompt skeleton.

This script addresses a reviewer-style baseline request without changing the
core prompt family: it reuses the frontier DeepSeek v3.2 title-recovery prompt
and few-shot exemplars, but swaps in the local Qwen2.5-7B model. It supports
two decoding modes:

1. plain beam decoding under the frontier-style prompt;
2. beam decoding with lightweight lexical constraints derived from dictionary
   glosses (up to two content constraints plus one title-suffix constraint).

The constraint heuristic is intentionally conservative. It only uses short,
high-confidence Chinese chunks that can be extracted deterministically from the
existing dictionary glosses, and it backs off automatically if a constraint set
is infeasible for generation.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


EXPERIMENTS_DIR = Path(__file__).resolve().parent
ROOT_DIR = EXPERIMENTS_DIR.parent
sys.path.insert(0, str(EXPERIMENTS_DIR))
sys.path.insert(0, str(ROOT_DIR))

from frontier_openrouter_dict import (  # noqa: E402
    SYSTEM_PROMPT,
    USER_TEMPLATE,
    build_fewshot_block,
    build_glosses_text,
    load_jsonl,
    normalize_prediction,
)
from src.dictionary_utils import BilingualDictionary  # noqa: E402


TITLE_SUFFIXES = set("經論記疏頌儀義傳錄贊序字品文觀根次門")
NUMERIC_CHARS = set("零〇一二三四五六七八九十百千萬兩")
CONSTRAINED_BEAM_SEARCH_PATH = ROOT_DIR / "third_party" / "hf_constrained_beam_search"

TRADITIONAL_MAP = str.maketrans(
    {
        "说": "説",
        "圣": "聖",
        "萨": "薩",
        "经": "經",
        "论": "論",
        "记": "記",
        "颂": "頌",
        "仪": "儀",
        "义": "義",
        "传": "傳",
        "录": "錄",
        "赞": "贊",
        "观": "觀",
        "门": "門",
        "罗": "羅",
        "启": "啟",
        "显": "顯",
        "华": "華",
        "怀": "懷",
        "贞": "貞",
        "数": "數",
        "围": "圍",
    }
)

BRACKET_RE = re.compile(r"【([^】]+)】")
TITLE_RE = re.compile(r"《([^》]+)》")
CHINESE_RE = re.compile(r"[\u4e00-\u9fff]+")
NOISE_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|（[^）]*）")
ALTERNATIVE_CUE_RE = re.compile(r"[、；;/]|[0-9]|具有|表示|之义|之義|结合|結合|如果|以及|或者|并|並")

GENERIC_SINGLETONS = {
    "其",
    "于",
    "與",
    "为",
    "為",
    "做",
    "使",
    "全",
    "都",
    "共",
    "同",
    "此",
    "彼",
    "所",
    "之",
    "者",
}

BAD_CONTENT_PHRASES = {
    "次序",
    "印信",
    "印信烙",
    "發語詞",
    "发语词",
    "範圍",
    "范围",
    "全都",
    "共同",
    "頌偈",
    "女性",
    "巧女",
}

BAD_CONTENT_TAILS = {"序", "詞", "词", "類", "类", "性", "烙"}


def to_traditional(text: str) -> str:
    return text.translate(TRADITIONAL_MAP)


def iter_chinese_chunks(text: str) -> list[str]:
    return CHINESE_RE.findall(text)


def normalize_gloss_text(text: str) -> str:
    text = to_traditional(text or "")
    text = NOISE_RE.sub("", text)
    text = BRACKET_RE.sub("", text)
    return text.strip()


def extract_title_suffix(explanation: str) -> str | None:
    trad = to_traditional(explanation or "")

    suffix_candidates: list[str] = []
    for title in TITLE_RE.findall(trad):
        for ch in reversed(title):
            if ch in TITLE_SUFFIXES:
                suffix_candidates.append(ch)
                break

    if "佛經" in trad or "佛经" in trad:
        suffix_candidates.append("經")

    clean = normalize_gloss_text(trad)
    for chunk in iter_chinese_chunks(clean):
        for ch in chunk:
            if ch in TITLE_SUFFIXES:
                suffix_candidates.append(ch)
                break

    return suffix_candidates[-1] if suffix_candidates else None


def extract_content_candidates(explanation: str) -> list[tuple[str, int]]:
    trad = to_traditional(explanation or "")
    clean = normalize_gloss_text(trad)
    candidates: list[tuple[str, int]] = []

    bracket_chunks = ["".join(iter_chinese_chunks(to_traditional(raw))) for raw in BRACKET_RE.findall(trad)]
    bracket_chunks = [chunk for chunk in bracket_chunks if chunk]
    outside_plain = normalize_gloss_text(trad)

    # A single bracketed gloss with no competing outside text is often a preferred lexeme.
    if len(bracket_chunks) == 1 and not outside_plain:
        chunk = bracket_chunks[0]
        if 1 <= len(chunk) <= 2 and chunk not in GENERIC_SINGLETONS:
            candidates.append((chunk, 30))

    plain = TITLE_RE.sub("", clean)
    chunks = iter_chinese_chunks(plain)
    if not chunks:
        return candidates

    if len(chunks) == 1 and not ALTERNATIVE_CUE_RE.search(plain):
        chunk = chunks[0]
        if 1 <= len(chunk) <= 2:
            if (
                chunk not in GENERIC_SINGLETONS
                and chunk not in BAD_CONTENT_PHRASES
                and chunk[-1] not in BAD_CONTENT_TAILS
            ):
                score = 24 if len(chunk) == 2 else 18
                if all(ch in NUMERIC_CHARS for ch in chunk):
                    score += 3
                candidates.append((chunk, score))

    return candidates


def build_allowed_phrase_set(real_corpus_path: Path, exclude_inputs: set[str]) -> set[str]:
    allowed: set[str] = set()
    for row in load_jsonl(real_corpus_path):
        if row.get("input") in exclude_inputs:
            continue
        output = to_traditional(row.get("output", ""))
        for n in (1, 2):
            for start in range(0, max(0, len(output) - n + 1)):
                chunk = output[start : start + n]
                if len(chunk) != n:
                    continue
                if all("\u4e00" <= ch <= "\u9fff" for ch in chunk):
                    allowed.add(chunk)
    return allowed


def build_constraint_phrases(
    tangut_text: str,
    dictionary: BilingualDictionary,
    max_content_constraints: int,
    allowed_phrases: set[str],
) -> tuple[list[str], dict[str, object]]:
    glosses = dictionary.get_glosses(tangut_text)
    suffix_hits: list[tuple[int, str]] = []
    scored_content: list[tuple[int, int, str]] = []

    for idx, (_, entry) in enumerate(glosses):
        if not entry or not entry.explanationCN:
            continue

        suffix = extract_title_suffix(entry.explanationCN)
        if suffix:
            suffix_hits.append((idx, suffix))

        for candidate, base_score in extract_content_candidates(entry.explanationCN):
            if not all(ch in NUMERIC_CHARS for ch in candidate) and candidate not in allowed_phrases:
                continue
            if len(candidate) == 1 and candidate in TITLE_SUFFIXES:
                continue
            score = base_score + min(len(candidate), 2) * 4 - idx
            scored_content.append((score, idx, candidate))

    suffix_phrase = suffix_hits[-1][1] if suffix_hits else None

    selected: list[str] = []
    for _, _, candidate in sorted(scored_content, key=lambda item: (-item[0], item[1], item[2])):
        if candidate in selected:
            continue
        if any(candidate in chosen or chosen in candidate for chosen in selected):
            continue
        selected.append(candidate)
        if len(selected) >= max_content_constraints:
            break

    phrases = selected.copy()
    if suffix_phrase and suffix_phrase not in phrases:
        phrases.append(suffix_phrase)

    debug = {
        "content_constraints": selected,
        "suffix_constraint": suffix_phrase,
        "num_gloss_segments": len(glosses),
    }
    return phrases, debug


def build_generation_attempts(phrases: list[str]) -> list[list[str]]:
    if not phrases:
        return [[]]

    attempts = [phrases]
    if len(phrases) > 1:
        attempts.append([phrases[-1]])
    attempts.append([])

    unique_attempts: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for attempt in attempts:
        key = tuple(attempt)
        if key not in seen:
            unique_attempts.append(attempt)
            seen.add(key)
    return unique_attempts


def generate_prediction(
    *,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    num_beams: int,
    length_penalty: float,
    phrases: list[str],
):
    import torch

    device = next(model.parameters()).device
    batch = tokenizer(prompt, return_tensors="pt")
    batch = {key: value.to(device) for key, value in batch.items()}
    input_length = batch["input_ids"].shape[1]

    force_words_ids = []
    for phrase in phrases:
        token_ids = tokenizer.encode(phrase, add_special_tokens=False)
        if token_ids:
            force_words_ids.append(token_ids)

    generation_kwargs = {
        "do_sample": False,
        "num_beams": max(num_beams, 1),
        "length_penalty": length_penalty,
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if force_words_ids:
        generation_kwargs["force_words_ids"] = force_words_ids
        generation_kwargs["num_beams"] = max(num_beams, len(force_words_ids) * 2, 4)
        generation_kwargs["early_stopping"] = True
        generation_kwargs["custom_generate"] = str(CONSTRAINED_BEAM_SEARCH_PATH)
        generation_kwargs["trust_remote_code"] = True

    with torch.inference_mode():
        generated = model.generate(**batch, **generation_kwargs)

    completion = generated[0][input_length:]
    return tokenizer.decode(completion, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run local Qwen decoding under the frontier DeepSeek prompt with optional lexical constraints."
    )
    parser.add_argument("--test-set", default="data/eval/test_set.jsonl")
    parser.add_argument("--dict-path", default="data/dictionary/dictionary.json")
    parser.add_argument("--dev-set", default="data/eval/dev_set.jsonl")
    parser.add_argument("--model-path", default="models/qwen2.5-7b-instruct")
    parser.add_argument("--real-corpus", default="data/raw/tangut_output.jsonl")
    parser.add_argument("--output", required=True)
    parser.add_argument("--method-name", default="local_qwen_frontier_style")
    parser.add_argument("--prompt-mode", choices=["strict", "fewshot"], default="fewshot")
    parser.add_argument("--constraint-mode", choices=["none", "lexical"], default="none")
    parser.add_argument("--max-content-constraints", type=int, default=1)
    parser.add_argument("--num-beams", type=int, default=8)
    parser.add_argument("--length-penalty", type=float, default=0.8)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    args = parser.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dictionary = BilingualDictionary(args.dict_path)
    test_rows = load_jsonl(Path(args.test_set))
    if args.start_index:
        test_rows = test_rows[args.start_index :]
    if args.limit is not None:
        test_rows = test_rows[: args.limit]
    allowed_phrases = build_allowed_phrase_set(
        Path(args.real_corpus),
        exclude_inputs={row["input"] for row in test_rows},
    )

    fewshot_block = ""
    if args.prompt_mode == "fewshot":
        dev_rows = load_jsonl(Path(args.dev_set))
        fewshot_block = (
            "下面是若干同任务示例。注意这些示例的共同规律：结果保持短标题体，"
            "不会把词典里的泛化佛经名机械照抄成最终题名。\n\n"
            f"{build_fewshot_block(dev_rows, dictionary)}\n\n"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(test_rows, start=1):
            glosses = build_glosses_text(item["input"], dictionary)
            user_prompt = USER_TEMPLATE.format(
                tangut_text=item["input"],
                glosses=glosses,
            )
            if fewshot_block:
                user_prompt = fewshot_block + user_prompt
            prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{user_prompt}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

            requested_phrases = []
            constraint_debug: dict[str, object] = {}
            if args.constraint_mode == "lexical":
                requested_phrases, constraint_debug = build_constraint_phrases(
                    item["input"],
                    dictionary,
                    max_content_constraints=args.max_content_constraints,
                    allowed_phrases=allowed_phrases,
                )
                attempts = build_generation_attempts(requested_phrases)
            else:
                attempts = [[]]

            raw_prediction = ""
            applied_phrases: list[str] = []
            notes = ""
            for attempt_idx, phrases in enumerate(attempts, start=1):
                try:
                    raw_prediction = generate_prediction(
                        model=model,
                        tokenizer=tokenizer,
                        prompt=prompt,
                        max_new_tokens=args.max_new_tokens,
                        num_beams=args.num_beams,
                        length_penalty=args.length_penalty,
                        phrases=phrases,
                    )
                    applied_phrases = phrases
                    if attempt_idx > 1:
                        notes = f"fallback_attempt={attempt_idx}"
                    break
                except Exception as exc:  # noqa: BLE001
                    notes = f"generation_error={type(exc).__name__}:{exc}"
                    raw_prediction = ""

            final_title = normalize_prediction(raw_prediction)
            record = {
                "input": item["input"],
                "reference": item["output"],
                "prediction": final_title,
                "glosses": glosses,
                "method": args.method_name,
                "prompt_variant": f"{args.prompt_mode}_title_only_frontier_style",
                "constraint_mode": args.constraint_mode,
                "requested_constraints": requested_phrases,
                "applied_constraints": applied_phrases,
                "constraint_debug": constraint_debug,
                "raw_response": raw_prediction,
                "notes": notes,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            print(
                f"[{idx}/{len(test_rows)}] constraints={applied_phrases or 'none'} -> {final_title}"
            )


if __name__ == "__main__":
    main()
