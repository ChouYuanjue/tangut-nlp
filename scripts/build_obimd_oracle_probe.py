#!/usr/bin/env python3
"""Build a small real oracle probe from OBIMD and map it into EVOBC IDs.

The resulting examples are real OBIMD sentence-level reading sequences, but
their source-side symbols are canonicalized into the EVOBC inventory through
the shared modern-character labels exposed by the two resources.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.oracle_surrogate_utils import encode_id_sequence  # noqa: E402


DEFAULT_INSTRUCTION = "请将以下甲骨文符号序列翻译为现代中文："
GROUP_PATTERN = re.compile(r"^InscriptionSentence")


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_repo_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def dump_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dump_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_candidates(
    data: list[dict],
    main_char_meta: dict,
    char_to_ids: dict[str, list[str]],
    id_to_surrogate: dict[str, str],
    instruction: str,
    min_len: int,
    max_len: int,
) -> tuple[list[dict], dict]:
    stats = Counter()
    deduped: dict[tuple[str, ...], dict] = {}

    for item in data:
        rubbing_name = item.get("RubbingName", "")
        for group in item.get("RecordUtilSentenceGroupVoList", []):
            group_category = group.get("GroupCategory", "")
            if not GROUP_PATTERN.match(group_category):
                stats["skip_non_sentence"] += 1
                continue

            chars = sorted(
                group.get("RecordUtilOracleCharVoList", []),
                key=lambda row: row.get("OrderNumber", 0),
            )
            stats["sentence_groups_seen"] += 1

            input_ids: list[str] = []
            reference_chars: list[str] = []
            obimd_main_uids: list[str] = []
            ok = True
            fail_reason = None

            for ch in chars:
                uid = ch.get("Label", "")
                meta = main_char_meta.get(uid)
                if meta is None:
                    ok = False
                    fail_reason = "missing_uid"
                    break

                modern_char = meta.get("codepoint", "")
                candidate_ids = char_to_ids.get(modern_char)
                if not candidate_ids:
                    ok = False
                    fail_reason = "not_in_evo"
                    break

                chosen_id = candidate_ids[0]
                input_ids.append(chosen_id)
                reference_chars.append(modern_char)
                obimd_main_uids.append(uid)

            if not ok:
                stats[fail_reason] += 1
                continue

            if not (min_len <= len(input_ids) <= max_len):
                stats["skip_length"] += 1
                continue

            key = tuple(input_ids)
            if key in deduped:
                stats["dedup_collisions"] += 1
                continue

            raw_input = " ".join(input_ids)
            encoded_input = encode_id_sequence(input_ids, id_to_surrogate, strict=True)
            reference = "".join(reference_chars)

            deduped[key] = {
                "instruction": instruction,
                "input": raw_input,
                "output": reference,
                "metadata": {
                    "source_repo": "OBIMD",
                    "canonicalized_to": "EVOBC IDs via shared modern-character labels",
                    "rubbing_name": rubbing_name,
                    "group_category": group_category,
                    "sentence_length": len(input_ids),
                    "obimd_main_uids": obimd_main_uids,
                },
                "_encoded_input": encoded_input,
                "_input_ids": input_ids,
            }
            stats["kept_unique"] += 1

    candidates = list(deduped.values())
    stats["length_distribution"] = dict(
        sorted(Counter(row["metadata"]["sentence_length"] for row in candidates).items())
    )
    return candidates, dict(stats)


def stratified_take(
    buckets: dict[int, list[dict]],
    target_size: int,
) -> list[dict]:
    selected: list[dict] = []
    ordered_lengths = sorted(buckets)
    while len(selected) < target_size:
        moved = False
        for length in ordered_lengths:
            bucket = buckets[length]
            if not bucket:
                continue
            selected.append(bucket.pop())
            moved = True
            if len(selected) >= target_size:
                break
        if not moved:
            break
    return selected


def split_candidates(
    candidates: list[dict],
    seed: int,
    train_size: int,
    dev_size: int,
    test_size: int,
) -> dict[str, list[dict]]:
    buckets: dict[int, list[dict]] = defaultdict(list)
    for row in candidates:
        buckets[row["metadata"]["sentence_length"]].append(row)

    rng = random.Random(seed)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    train_rows = stratified_take(buckets, train_size)
    dev_rows = stratified_take(buckets, dev_size)
    test_rows = stratified_take(buckets, test_size)
    remainder = sum(len(bucket) for bucket in buckets.values())

    return {
        "train": train_rows,
        "dev": dev_rows,
        "test": test_rows,
        "_remainder": remainder,
    }


def materialize_encoded(rows: list[dict]) -> list[dict]:
    encoded_rows: list[dict] = []
    for row in rows:
        metadata = dict(row["metadata"])
        encoded_rows.append(
            {
                "instruction": row["instruction"],
                "input": row["_encoded_input"],
                "output": row["output"],
                "metadata": metadata,
                "input_ids": row["_input_ids"],
            }
        )
    return encoded_rows


def strip_internal_fields(rows: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for row in rows:
        cleaned.append(
            {
                "instruction": row["instruction"],
                "input": row["input"],
                "output": row["output"],
                "metadata": dict(row["metadata"]),
            }
        )
    return cleaned


def split_summary(rows: list[dict]) -> dict:
    lengths = Counter(row["metadata"]["sentence_length"] for row in rows)
    return {
        "count": len(rows),
        "length_distribution": dict(sorted(lengths.items())),
        "examples": [
            {
                "input": row["input"],
                "output": row["output"],
                "rubbing_name": row["metadata"]["rubbing_name"],
                "group_category": row["metadata"]["group_category"],
            }
            for row in rows[:5]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an OBIMD-derived real oracle probe in EVOBC ID space."
    )
    parser.add_argument("--obimd-root", default="/tmp/OBIMD-hf")
    parser.add_argument(
        "--obimd-data",
        default=None,
        help="Override OBIMD data.json path. Defaults to <obimd-root>/data.json.",
    )
    parser.add_argument(
        "--main-char-json",
        default=None,
        help="Override Main-character.json path. Defaults to the OBIMD supplement path.",
    )
    parser.add_argument(
        "--char-to-ids",
        default="data/oracle_evo/assets/char_to_ids.json",
        help="Path to the generated EVOBC char_to_ids map.",
    )
    parser.add_argument(
        "--id-to-surrogate",
        default="data/oracle_evo/assets/id_to_surrogate.json",
        help="Path to the generated EVOBC id_to_surrogate map.",
    )
    parser.add_argument("--output-root", default="data/oracle_evo")
    parser.add_argument("--output-prefix", default="obimd_canonicalized")
    parser.add_argument("--train-size", type=int, default=50)
    parser.add_argument("--dev-size", type=int, default=50)
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--min-len", type=int, default=4)
    parser.add_argument("--max-len", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    args = parser.parse_args()

    obimd_root = Path(args.obimd_root)
    obimd_data_path = Path(args.obimd_data or obimd_root / "data.json")
    main_char_path = Path(
        args.main_char_json
        or obimd_root / "Hierarchical Character Metadata Supplement/Main-character.json"
    )
    output_root = resolve_repo_path(args.output_root)

    data = load_json(obimd_data_path)
    main_char_meta = load_json(main_char_path)
    char_to_ids = load_json(resolve_repo_path(args.char_to_ids))
    id_to_surrogate = load_json(resolve_repo_path(args.id_to_surrogate))

    candidates, candidate_stats = build_candidates(
        data=data,
        main_char_meta=main_char_meta,
        char_to_ids=char_to_ids,
        id_to_surrogate=id_to_surrogate,
        instruction=args.instruction,
        min_len=args.min_len,
        max_len=args.max_len,
    )
    splits = split_candidates(
        candidates=candidates,
        seed=args.seed,
        train_size=args.train_size,
        dev_size=args.dev_size,
        test_size=args.test_size,
    )

    summary = {
        "source_repo": "OBIMD",
        "canonicalization": "OBIMD Main-character.json codepoint -> EVOBC char_to_ids -> EVOBC IDs",
        "instruction": args.instruction,
        "seed": args.seed,
        "candidate_stats": candidate_stats,
        "train": split_summary(splits["train"]),
        "dev": split_summary(splits["dev"]),
        "test": split_summary(splits["test"]),
        "remaining_unique_candidates_after_split": splits["_remainder"],
    }

    for split_name in ("train", "dev", "test"):
        raw_rows = strip_internal_fields(splits[split_name])
        encoded_rows = materialize_encoded(splits[split_name])

        raw_path = output_root / "raw" / f"{args.output_prefix}_{split_name}.jsonl"
        encoded_path = output_root / "eval" / f"{args.output_prefix}_{split_name}_encoded.jsonl"
        dump_jsonl(raw_path, raw_rows)
        dump_jsonl(encoded_path, encoded_rows)

    summary_path = output_root / "raw" / f"{args.output_prefix}_summary.json"
    dump_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
