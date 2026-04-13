#!/usr/bin/env python3
"""Rewrite JSONL oracle ID sequences into reversible surrogate strings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.oracle_surrogate_utils import (  # noqa: E402
    encode_id_sequence,
    parse_id_sequence,
)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON dict in {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode oracle ID sequences in a JSONL file.")
    parser.add_argument("--input", required=True, help="Path to input JSONL.")
    parser.add_argument("--output", required=True, help="Path to output JSONL.")
    parser.add_argument(
        "--mapping",
        default="data/oracle_evo/assets/id_to_surrogate.json",
        help="Path to the generated id_to_surrogate.json map.",
    )
    parser.add_argument(
        "--field",
        default="input",
        help="JSON field containing the raw oracle ID sequence.",
    )
    parser.add_argument(
        "--field-out",
        default=None,
        help="Destination field for the surrogate string. Defaults to overwriting --field.",
    )
    parser.add_argument(
        "--keep-original-field",
        default="input_ids",
        help="Optional field to store the parsed canonical IDs. Use an empty string to disable.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail on unknown IDs instead of writing [UNK].",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    mapping = load_json(Path(args.mapping))
    field_out = args.field_out or args.field
    keep_original = args.keep_original_field.strip()

    encoded_rows = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8") as f_in, output_path.open(
        "w",
        encoding="utf-8",
    ) as f_out:
        for line_number, line in enumerate(f_in, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if args.field not in row:
                raise KeyError(f"Missing field {args.field!r} at line {line_number}")

            symbol_ids = parse_id_sequence(row[args.field])
            encoded = encode_id_sequence(symbol_ids, mapping, strict=args.strict)
            if keep_original:
                row[keep_original] = symbol_ids
            row[field_out] = encoded
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")
            encoded_rows += 1

    print(
        json.dumps(
            {
                "status": "ok",
                "rows_encoded": encoded_rows,
                "output": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

