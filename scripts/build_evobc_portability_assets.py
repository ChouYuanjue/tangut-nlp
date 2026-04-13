#!/usr/bin/env python3
"""Build EVOBC-based portability assets for the Tangut training pipeline.

This adapter keeps the external EVOBC IDs auditable while emitting a Tangut-
style dictionary whose source-side symbols are reversible single-codepoint
surrogates. The output can be fed directly into the existing dictionary-
grounded prompting, synthetic SFT, and DPO scripts without modifying the
core Tangut code path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.oracle_surrogate_utils import (  # noqa: E402
    DEFAULT_SURROGATE_START,
    build_char_to_ids,
    build_reward_dict,
    build_surrogate_tables,
    build_tangut_style_dictionary,
    load_id_to_char,
)


DEFAULT_INPUT_CANDIDATES = [
    REPO_ROOT / "data/oracle_evo/raw/Key&Value.json",
    Path("/tmp/character-Evolution-Dataset/Key&Value.json"),
]


def resolve_input_path(explicit_path: str | None) -> Path:
    if explicit_path:
        path = Path(explicit_path).expanduser()
        if path.exists():
            return path
        raise FileNotFoundError(f"EVOBC mapping file not found: {path}")

    for candidate in DEFAULT_INPUT_CANDIDATES:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not locate Key&Value.json automatically. "
        "Pass --evobc-kv or place the file in data/oracle_evo/raw/."
    )


def dump_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_metadata(
    *,
    input_path: Path,
    id_to_char: dict[str, str],
    char_to_ids: dict[str, list[str]],
    id_to_surrogate: dict[str, str],
    start_codepoint: int,
) -> dict:
    duplicate_char_count = sum(1 for ids in char_to_ids.values() if len(ids) > 1)
    preview_ids = sorted(id_to_char)[:5]
    preview = [
        {
            "external_id": symbol_id,
            "modern_char": id_to_char[symbol_id],
            "surrogate_codepoint": f"U+{ord(id_to_surrogate[symbol_id]):05X}",
        }
        for symbol_id in preview_ids
    ]
    return {
        "source": "EVOBC",
        "input_mapping_file": str(input_path),
        "num_ids": len(id_to_char),
        "num_unique_chars": len(char_to_ids),
        "duplicate_char_count": duplicate_char_count,
        "surrogate_start_codepoint": f"U+{start_codepoint:05X}",
        "preview": preview,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EVOBC portability assets.")
    parser.add_argument(
        "--evobc-kv",
        default=None,
        help="Path to EVOBC Key&Value.json. If omitted, the script searches common local paths.",
    )
    parser.add_argument(
        "--output-root",
        default="data/oracle_evo",
        help="Root directory for generated assets.",
    )
    parser.add_argument(
        "--start-codepoint",
        type=lambda x: int(x, 0),
        default=DEFAULT_SURROGATE_START,
        help="Starting Unicode codepoint for surrogate assignment (default: 0xF0000).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print summary only; do not write files.",
    )
    args = parser.parse_args()

    input_path = resolve_input_path(args.evobc_kv)
    id_to_char = load_id_to_char(input_path)
    char_to_ids = build_char_to_ids(id_to_char)
    id_to_surrogate, surrogate_to_id = build_surrogate_tables(
        id_to_char,
        start_codepoint=args.start_codepoint,
    )
    dictionary_entries = build_tangut_style_dictionary(id_to_char, id_to_surrogate)
    reward_dict = build_reward_dict(id_to_char, id_to_surrogate)
    metadata = build_metadata(
        input_path=input_path,
        id_to_char=id_to_char,
        char_to_ids=char_to_ids,
        id_to_surrogate=id_to_surrogate,
        start_codepoint=args.start_codepoint,
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if args.dry_run:
        return

    output_root = Path(args.output_root)
    dump_json(output_root / "assets/id_to_char.json", id_to_char)
    dump_json(output_root / "assets/char_to_ids.json", char_to_ids)
    dump_json(output_root / "assets/id_to_surrogate.json", id_to_surrogate)
    dump_json(output_root / "assets/surrogate_to_id.json", surrogate_to_id)
    dump_json(output_root / "assets/metadata.json", metadata)
    dump_json(output_root / "dictionary/evo_dictionary.json", dictionary_entries)
    dump_json(output_root / "dictionary/evo_reward_dict.json", reward_dict)

    print(
        json.dumps(
            {
                "status": "ok",
                "output_root": str(output_root),
                "dictionary_entries": len(dictionary_entries),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

